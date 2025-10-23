import os
import omegaconf
import hydra
import copy
import torch
VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]


    
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

def load_r3m(foldername, model_name, device):
    modelpath = os.path.join(foldername, f"{model_name}.pt")
    configpath = os.path.join( foldername, f"{model_name}.yaml")
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg, device)
    rep = hydra.utils.instantiate(cleancfg)

    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise ValueError(f"Device should be a string or torch.device, got {type(device)}")

    if device.type == 'cuda':
        torch.cuda.set_device(0)
        rep = rep.to(device)
        rep = torch.nn.DataParallel(rep, device_ids=[0])
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])

    rep.load_state_dict(r3m_state_dict) # resnet18, resnet34
    return rep