import copy
import torch
from libero.lifelong.algos.base import Sequential
from torchvision import transforms
from libero.libero.utils.load_r3m import load_r3m  # type: ignore
import torch.nn as nn

class SingleTask(Sequential):
    """
    The sequential BC baseline.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        # self.init_pi = copy.deepcopy(self.policy)
        self.transform_size = transforms.CenterCrop(224)#
        # Load the encoder
        self.encoder = load_r3m(self.cfg.r3m_folder, self.cfg.model_name, cfg.device)
        self.encoder.train() 

    def start_task(self, task):
        # re-initialize every new task
        # self.policy = copy.deepcopy(self.init_pi)
        self.policy.language_encoder.set_dataset_id(self.current_task)
        super().start_task(task)

    def get_im(self, obs):
        view_agent = obs['agentview_rgb']
        view_hand = obs['eye_in_hand_rgb']
        view_agent = torch.flip(view_agent, [3])  
        view_hand = torch.flip(view_hand, [3])    
        img = torch.cat([
            torch.cat([view_agent[:, 0, :, :, :], view_agent[:, 1, :, :, :]], dim=3),   
            torch.cat([view_hand[:, 0, :, :, :], view_hand[:, 1, :, :, :]], dim=3)],
            dim=2)
        return img
    
    
    def forward(self, data):
        # with torch.no_grad():
        img = self.get_im(data['obs'])
        img_resized = self.transform_size(img)
        z = self.encoder(img_resized * 255)
        extra_lang = self.get_extra_lang(data)
        r_z = torch.cat([z, extra_lang], -1)
        
        return r_z

    def get_extra_lang(self, data):
        extra_lang = self.policy.object_encoder(data)
        return extra_lang
    
    def observe(self, data):

        data = self.map_tensor_to_device(data)
        # with torch.no_grad():
        current_features = self.forward(data)

        self.optimizer.zero_grad()
        data['feature'] = current_features
        
        # model_loss
        loss_model = self.policy.compute_loss(data)
        self.model_loss = loss_model.item()
        loss = loss_model 
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()
    
    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            data['feature'] = self.forward(data)
            loss_model = self.policy.compute_loss(data)
            self.model_loss = loss_model.item()
            loss = loss_model 
        return loss.item()
    
    def end_task(self, dataset, task_id, benchmark, env=None):
        self.freeze_model(self.encoder)
        self.verify_frozen_state(self.encoder)