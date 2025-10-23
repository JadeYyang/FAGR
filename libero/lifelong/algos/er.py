import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *
from r3m import load_r3m 
from torchvision import transforms
def cycle(dl):
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class ER(Sequential):
    """
    The experience replay policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # we truncate each sequence dataset to a buffer, when replay is used,
        # concate all buffers to form a single replay buffer for replay.
        self.datasets = []
        self.descriptions = []
        self.buffer = None
        self.encoder = load_r3m(self.cfg.model_name)
        self.encoder.train() 
        self.encoder.to(cfg.device)
        self.transform_size = transforms.CenterCrop(224)#

    def start_task(self, task):
        super().start_task(task)
        self.policy.language_encoder.set_dataset_id(self.current_task)
        if self.current_task > 0:
            self.freeze_model(self.encoder)
            self.verify_frozen_state(self.encoder)
            # WARNING: currently we have a fixed size memory for each task.
            buffers = [
                TruncatedSequenceDataset(dataset, self.cfg.lifelong.n_memories)
                for dataset in self.datasets
            ]
            # [encoder['encoder'].eval() for encoder in self.policy.image_encoders.values()]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=False,
                )
            )

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
        # r_z = F.normalize(r_z, p=2, dim=1)
        
        return r_z

    def get_extra_lang(self, data):
        extra_lang = self.policy.object_encoder(data)
        return extra_lang

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)
        data = self.map_tensor_to_device(data)
        data['feature'] = self.forward(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        data['feature'] = self.forward(data)
        with torch.no_grad():
            loss = self.policy.compute_loss(data)
        return loss.item()