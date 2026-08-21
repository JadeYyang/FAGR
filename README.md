# FAGR: Feature-Action Generative Replay for Robotic Lifelong Imitation Learning


## Abstract

Conventional imitation learning (IL) struggles with scalability and catastrophic forgetting in sequential task learning, prompting the need for Lifelong Imitation Learning (LIL) to enable sustainable knowledge accumulation. However, existing LIL approaches, which largely depend on replaying demonstration data or generating visual pseudo-samples, suffer from high storage demands and low sample fidelity.

To address these issues, we propose **Feature-Action Generative Replay (FAGR)**, a novel LIL framework for robotic manipulation. FAGR begins by performing multimodal feature extraction and feature-action joint clustering for each task to retain cluster statistics, thereby dramatically reducing storage requirements by replacing raw demonstration data with compact cluster statistics. Subsequently, FAGR generates pseudo-feature samples from these statistics and infers corresponding actions via previously learned policy network, thereby obtaining feature-action pairs for past tasks. To enhance the reliability of these samples, FAGR incorporates a filtering mechanism based on action distribution confidence, which selectively retains the high-fidelity generative samples. These curated samples are then integrated with current-task data for joint optimization, enabling effective robust knowledge retention and transfer. 

To the best of our knowledge, **this work is the first to explore generative replay in the feature-action space for LIL**. Extensive experiments demonstrate that FAGR outperforms state-of-the-art methods in mitigating forgetting while maintaining high task success rates, thus achieving better stability-plasticity balance.

---

## 🚀 Quick Start

This section assumes that LIBERO and R3M are already available. Dataset and
checkpoint locations may be supplied through the standard LIBERO config or as
Hydra overrides.

### 1. Run the paper configuration

```bash
CUDA_VISIBLE_DEVICES=0 python -m libero.lifelong.main \
  experiment=fagr_paper \
  benchmark_name=LIBERO_SPATIAL \
  seed=10000 \
  device=cuda:0
```

Change `benchmark_name` to `LIBERO_OBJECT`, `LIBERO_GOAL`, or `LIBERO_10` for
the other 10-task suites. The extended experiment on the first 24 LIBERO-90
tasks is selected with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m libero.lifelong.main \
  experiment=fagr_paper \
  benchmark_name=LIBERO_90 \
  max_tasks=24 \
  seed=10000 \
  device=cuda:0
```

Evaluate a saved FAGR checkpoint independently with FAGR as the lifelong
algorithm and Diffusion BC as its policy:

```bash
CUDA_VISIBLE_DEVICES=0 python -m libero.lifelong.evaluate \
  --benchmark libero_spatial \
  --algo fagr \
  --policy bc_diffusion_policy \
  --seed 10000 \
  --load_task 9 \
  --task_id 0 \
  --checkpoint best \
  --device_id 0
```

`--load_task` selects the checkpoint after a lifelong task, while `--task_id`
selects the benchmark task to evaluate. Use `--run_id N` to avoid implicitly
selecting the latest `run_NNN` directory.

Run three separately recorded seeds for the paper's mean/std protocol and keep
the generated `config.json` with every result. The paper does not list the
three numerical seed values; `10000`, `20000`, and `30000` are reproducible
defaults for a new reproduction, not a claim about the authors' original runs.

### 2. Customize the YAML

Copy the paper configuration and edit the copy:

```bash
cp libero/configs/experiment/fagr_paper.yaml \
   libero/configs/experiment/fagr_custom.yaml
```

Then launch it by config name:

```bash
CUDA_VISIBLE_DEVICES=0 python -m libero.lifelong.main \
  experiment=fagr_custom \
  device=cuda:0
```

The main FAGR parameters are all exposed in that YAML:

- `lifelong.sample_size`: number of clusters per task (`K=16` in the paper)
- `lifelong.shrinkage_factor`: pseudo-feature scale (`lambda=0.75`)
- `lifelong.confidence_threshold`: Mahalanobis threshold (`tau=3`)
- `lifelong.replay_pool_size_per_task`: target accepted replay samples per past task
- `policy.down_dims`: Diffusion Policy U-Net widths (`[128, 256, 512]`)
- `train.warmup_epochs`: warm-up duration before cosine annealing
- `data.task_order_index`: LIBERO task order
- `max_tasks`: optional task-sequence truncation

The paper states that warm-up is used but does not report its exact duration;
the public paper configuration sets `warmup_epochs: 5` explicitly so every run
records the chosen value rather than hiding it in code.

Hydra command-line overrides can be used without copying the file, for example:

```bash
python -m libero.lifelong.main \
  experiment=fagr_paper \
  lifelong.confidence_threshold=4.5 \
  lifelong.shrinkage_factor=1.75 \
  seed=20000
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{yang2025fagr,
  title={FAGR: Feature-Action Generative Replay for Robotic Lifelong Imitation Learning},
  author={Yang, Yushi and Nie, Xiangli and Liu, Chang},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaborations, please contact:
- **Yushi Yang**: yangyushi2023@ia.ac.cn

---
# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Acknowledgements
1. We would like to thank the authors of [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) for providing the datasets, environments and codebase for our experiments.
