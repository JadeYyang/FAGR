import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, repeat, reduce
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.modules.conditional_unet1d import ConditionalUnet1D
from typing import Dict, Callable, List


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class ExtraModalities:
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):

        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )
        assert self.extra_low_level_feature_dim > 0, "[error] no extra information"

    def __call__(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])
        x = torch.cat(tensor_list, dim=-1)
        return x

    def output_shape(self, input_shape, shape_meta):
        return (self.extra_low_level_feature_dim,)

class BCDiffusionPolicy(BasePolicy):
    def __init__(self, cfg, shape_meta, n_task=None):
        super().__init__(cfg, shape_meta, n_task)
        policy_cfg = cfg.policy
        obs_feature_dim = 0
        # 1. encode image
        if not cfg.use_r3m:
            image_embed_size = policy_cfg.image_embed_size
            self.image_encoders = {}
            for name in shape_meta["all_shapes"].keys():
                if "rgb" in name or "depth" in name:
                    kwargs = policy_cfg.image_encoder.network_kwargs
                    kwargs.input_shape = shape_meta["all_shapes"][name]
                    kwargs.output_size = image_embed_size
                    kwargs.language_dim = (
                        policy_cfg.language_encoder.network_kwargs.input_size
                    )
                    self.image_encoders[name] = {
                        "input_shape": shape_meta["all_shapes"][name],
                        "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                    }
                    obs_feature_dim += image_embed_size
            self.encoders = nn.ModuleList(
                [x["encoder"] for x in self.image_encoders.values()]
            )
        ### 2. encode language
        policy_cfg.language_encoder.network_kwargs.num_encoders = n_task
        text_embed_size = policy_cfg.text_embed_size
        policy_cfg.language_encoder.network_kwargs.output_size = text_embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalities(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
        )
        obs_feature_dim += self.extra_encoder.extra_low_level_feature_dim
        ### 4. create diffusion model
        # parse shape_meta

        action_dim = shape_meta['ac_dim']
        input_dim = action_dim
        if cfg.use_r3m:
            global_cond_dim = obs_feature_dim * policy_cfg.n_obs_steps + text_embed_size  + cfg.r3m_dim 
        else:
            global_cond_dim = obs_feature_dim * policy_cfg.n_obs_steps + text_embed_size
            
        model = ConditionalUnet1D(
            input_dim = input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=policy_cfg.down_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
            cond_predict_scale=policy_cfg.cond_predict_scale
        )
        self.model = model
        self.noise_scheduler = DDPMScheduler(**policy_cfg.noise_scheduler)
        self.horizon = policy_cfg.horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = policy_cfg.n_action_steps
        self.n_obs_steps = policy_cfg.n_obs_steps
        self.num_inference_steps = policy_cfg.num_inference_steps
    # ========= encode =========
    def object_encoder(self, data):
        encoded = []
        B = data['task_emb'].shape[0]
        if not self.cfg.use_r3m:
            for img_name in self.image_encoders.keys():
                x = data["obs"][img_name]
                B, T, C, H, W = x.shape
                e = self.image_encoders[img_name]["encoder"](
                    x.reshape(B * T, C, H, W),
                    langs=data["task_emb"]
                    .reshape(B, 1, -1)
                    .repeat(1, T, 1)
                    .reshape(B * T, -1),
                ).view(B, T, -1)
                encoded.append(e)

        # 2. add joint states, gripper info, etc.
        if "obs" in data:
            encoded.append(self.extra_encoder(data["obs"]))  # add (B, T, H_extra)
            encoded = torch.cat(encoded, -1).reshape(B, -1)  # (B, T, H_all)
        # 3. language encoding
        lang_h = self.language_encoder(data)  # (B, H)
        return torch.cat([encoded, lang_h], -1)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def get_action(self, data):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        data = dict_apply(data, lambda x: x.to(self.device, non_blocking=True)) 
        value = next(iter(data['obs'].values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        # build input
        device = self.device

        # handle different ways of passing observation
        local_cond = None
        if self.cfg.use_r3m: 
            # global_cond = torch.cat([self.object_encoder(data), data['pro_sim']], dim=-1)
            global_cond = data['feature']
        else:
            global_cond = self.object_encoder(data)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=value.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond
        )  
        action_pred = nsample[...,:Da]      
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        return action
    
    def compute_loss(self, data):
        batch_size = data['actions'].shape[0]
        trajectory = data['actions']       
        if self.cfg.use_r3m:
            global_cond = data['feature']
        else:
            global_cond = self.object_encoder(data)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def forward(self, data):
        loss = self.compute_loss(data)
        return loss