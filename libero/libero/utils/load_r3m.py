import os
import omegaconf
import hydra
import copy
import torch
VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]

MODEL_ALIASES = {
    "res_18": "resnet18",
    "res_34": "resnet34",
    "res_50": "resnet50",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
}


    
def cleanup_config(cfg, device):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "r3m.R3M"
    config["device"] = device
    
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent

def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict

def _normalize_model_name(model_name):
    try:
        return MODEL_ALIASES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported R3M model {model_name!r}; choose one of "
            f"{sorted(MODEL_ALIASES)}"
        ) from exc


def _strip_data_parallel_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _load_local_r3m(foldername, model_name, device):
    legacy_name = {
        "resnet18": "res_18",
        "resnet34": "res_34",
        "resnet50": "res_50",
    }[model_name]
    candidates = [
        (
            os.path.join(foldername, f"{legacy_name}.pt"),
            os.path.join(foldername, f"{legacy_name}.yaml"),
        ),
        (
            os.path.join(foldername, "model.pt"),
            os.path.join(foldername, "config.yaml"),
        ),
    ]
    modelpath, configpath = next(
        ((model, config) for model, config in candidates
         if os.path.isfile(model) and os.path.isfile(config)),
        (None, None),
    )
    if modelpath is None:
        expected = ", ".join(f"{model} + {config}" for model, config in candidates)
        raise FileNotFoundError(
            f"no local R3M checkpoint found in {foldername!r}; expected {expected}"
        )

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg, device)
    rep = hydra.utils.instantiate(cleancfg)
    checkpoint = torch.load(modelpath, map_location=device)
    state_dict = remove_language_head(checkpoint["r3m"])
    rep.load_state_dict(_strip_data_parallel_prefix(state_dict))
    return rep.to(device)


def load_r3m(foldername, model_name, device):
    """Load R3M locally or let the official package download and cache it."""
    device = torch.device(device)
    model_name = _normalize_model_name(model_name)

    if foldername:
        return _load_local_r3m(foldername, model_name, device)

    from r3m import load_r3m as load_official_r3m

    rep = load_official_r3m(model_name)
    if isinstance(rep, torch.nn.DataParallel):
        rep = rep.module
    return rep.to(device)
