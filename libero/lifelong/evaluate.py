import argparse
import os
from pathlib import Path

import torch
from easydict import EasyDict

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import (
    CustomSequenceVLDataset,
    SequenceVLDataset,
    get_dataset,
)
from libero.lifelong.main import get_task_embs
from libero.lifelong.metric import evaluate_loss, evaluate_one_task_success
from libero.lifelong.utils import control_seed, safe_device


BENCHMARK_MAP = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_90": "LIBERO_90",
}

ALGO_MAP = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
    # FAGR is the method name; FGRA is the registered implementation class.
    "fagr": "FGRA",
    "fgra": "FGRA",
}

POLICY_MAP = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
    "bc_diffusion_policy": "BCDiffusionPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved LIBERO policy")
    parser.add_argument("--experiment_dir", default="experiments")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_MAP))
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_MAP))
    parser.add_argument("--policy", required=True, choices=sorted(POLICY_MAP))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Select taskN_model.pth or taskN_last_model.pth.",
    )
    parser.add_argument("--run_id", type=int, help="Use a specific run_NNN directory")
    parser.add_argument("--n_eval", type=int, help="Override cfg.eval.n_eval")
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Reserved for compatibility; the shared evaluator does not save videos.",
    )
    args = parser.parse_args()

    if args.algo == "multitask" and args.ep is None:
        parser.error("--ep is required for multitask evaluation")
    if args.algo != "multitask" and args.load_task is None:
        parser.error("--load_task is required for lifelong evaluation")
    if args.algo in ("fagr", "fgra") and args.policy != "bc_diffusion_policy":
        parser.error("FAGR/FGRA must be evaluated with --policy bc_diffusion_policy")
    if args.n_eval is not None and args.n_eval <= 0:
        parser.error("--n_eval must be positive")
    return args


def find_run(experiment_dir, run_id=None):
    if run_id is not None:
        run_folder = experiment_dir / f"run_{run_id:03d}"
        if not run_folder.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_folder}")
        return run_folder

    run_folders = []
    for path in experiment_dir.glob("run_*"):
        if not path.is_dir():
            continue
        try:
            run_folders.append((int(path.name.split("run_", 1)[-1]), path))
        except ValueError:
            continue
    if not run_folders:
        raise FileNotFoundError(f"no run_NNN directory under {experiment_dir}")
    return max(run_folders)[1]


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        checkpoint = torch.load(path, map_location=device)
    if "state_dict" not in checkpoint or "cfg" not in checkpoint:
        raise ValueError(f"invalid LIBERO checkpoint: {path}")
    cfg = checkpoint["cfg"]
    if not isinstance(cfg, EasyDict):
        cfg = EasyDict(cfg)
    return checkpoint, cfg


def load_r3m_encoder(algo, checkpoint, run_folder, device):
    if not hasattr(algo, "encoder"):
        return

    encoder_state = checkpoint.get("encoder_state_dict")
    encoder_path = run_folder / "r3m_encoder.pth"
    if encoder_state is None and encoder_path.is_file():
        try:
            encoder_checkpoint = torch.load(
                encoder_path, map_location=device, weights_only=False
            )
        except TypeError:
            encoder_checkpoint = torch.load(encoder_path, map_location=device)
        encoder_state = encoder_checkpoint["state_dict"]

    if encoder_state is None:
        print(
            "[warning] checkpoint predates R3M encoder saving; evaluation will use "
            "the base R3M weights instead of the task-1 fine-tuned encoder"
        )
        return
    algo.encoder.load_state_dict(encoder_state)
    algo.freeze_model(algo.encoder)
    print("[ok] loaded the task-1 R3M encoder")


def main():
    args = parse_args()
    control_seed(args.seed)
    device = f"cuda:{args.device_id}"

    experiment_dir = (
        Path(args.experiment_dir)
        / BENCHMARK_MAP[args.benchmark]
        / ALGO_MAP[args.algo]
        / f"{POLICY_MAP[args.policy]}_seed{args.seed}"
    )
    run_folder = find_run(experiment_dir, args.run_id)

    if args.algo == "multitask":
        model_path = run_folder / f"multitask_model_ep{args.ep}.pth"
    else:
        suffix = "model" if args.checkpoint == "best" else "last_model"
        model_path = run_folder / f"task{args.load_task}_{suffix}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {model_path}")

    checkpoint, cfg = load_checkpoint(model_path, device)
    cfg.device = device
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    if args.n_eval is not None:
        cfg.eval.n_eval = args.n_eval

    expected_benchmark = BENCHMARK_MAP[args.benchmark]
    if cfg.benchmark_name != expected_benchmark:
        raise ValueError(
            f"checkpoint benchmark is {cfg.benchmark_name}, not {expected_benchmark}"
        )
    expected_policy = POLICY_MAP[args.policy]
    if cfg.policy.policy_type != expected_policy:
        raise ValueError(
            f"checkpoint policy is {cfg.policy.policy_type}, not {expected_policy}"
        )

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks
    if getattr(cfg, "max_tasks", None) is not None:
        n_manip_tasks = min(n_manip_tasks, int(cfg.max_tasks))
    group_size = int(cfg.data.task_group_size)
    n_tasks = n_manip_tasks // group_size

    if not 0 <= args.task_id < n_manip_tasks:
        raise ValueError(f"--task_id must be in [0, {n_manip_tasks - 1}]")
    if args.load_task is not None and not 0 <= args.load_task < n_tasks:
        raise ValueError(f"--load_task must be in [0, {n_tasks - 1}]")

    descriptions = [benchmark.get_task(i).language for i in range(n_manip_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    algo = safe_device(get_algo_class(ALGO_MAP[args.algo])(n_tasks, cfg), device)
    algo.policy.load_state_dict(checkpoint["state_dict"])
    load_r3m_encoder(algo, checkpoint, run_folder, device)

    language_encoder = getattr(algo.policy, "language_encoder", None)
    if getattr(language_encoder, "multi_encoder", False):
        language_encoder.set_dataset_id(args.task_id // group_size)

    task = benchmark.get_task(args.task_id)
    dataset_path = os.path.join(
        cfg.folder, benchmark.get_task_demonstration(args.task_id)
    )
    dataset, _ = get_dataset(
        dataset_path=dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    if cfg.policy.policy_type in ("BCDiffusionPolicy", "BCMeanFlowPolicy"):
        dataset = CustomSequenceVLDataset(dataset, task_embs[args.task_id], cfg)
    else:
        dataset = SequenceVLDataset(dataset, task_embs[args.task_id])

    algo.eval()
    test_loss = float(evaluate_loss(cfg, algo, benchmark, [dataset])[0])
    success_rate = evaluate_one_task_success(
        cfg=cfg,
        algo=algo,
        task=task,
        task_emb=benchmark.get_task_emb(args.task_id),
        task_id=args.task_id,
    )

    if args.save_videos:
        print("[warning] --save-videos is not implemented by the shared evaluator")
    save_dir = Path(f"{args.experiment_dir}_saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    load_label = f"ep{args.ep}" if args.algo == "multitask" else f"load{args.load_task}"
    save_path = save_dir / (
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_"
        f"{load_label}_on{args.task_id}.stats"
    )
    torch.save(
        {
            "loss": test_loss,
            "success_rate": success_rate,
            "checkpoint": str(model_path),
            "task_id": args.task_id,
        },
        save_path,
    )
    print(f"[info] checkpoint: {model_path}")
    print(f"[info] loss: {test_loss:.6f}")
    print(f"[info] success rate: {success_rate:.4f}")
    print(f"[info] results: {save_path}")


if __name__ == "__main__":
    main()
