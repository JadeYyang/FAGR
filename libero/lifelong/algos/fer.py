import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.fftpack import dct
from libero.lifelong.algos.base import Sequential
from r3m import load_r3m  
import numpy as np
from libero.lifelong.models import *
import random

class FER(Sequential):
    '''
    Feature replay
    '''
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.sample_size = self.cfg.lifelong.sample_size  # Number of representative samples(n)

        self.sample_size_per_task = self.cfg.lifelong.sample_size_per_task

        # Load the encoder
        self.encoder = load_r3m(self.cfg.model_name)
        self.encoder.train() 
        self.encoder.to(cfg.device)

        self.selected_features_per_task = [[] for _ in range(n_tasks)]
        self.selected_actions_per_task = [[] for _ in range(n_tasks)]
        self.use_cluster_sample = self.cfg.lifelong.use_cluster_sample
        self.use_random_sample = self.cfg.lifelong.use_random_sample
        # Temporary storage during the first epoch

        # Transformations for data augmentation (if needed)
        self.transform_size = transforms.CenterCrop(224)#

        self.model_loss = 0


    def start_task(self, task):
        super().start_task(task)
        self.policy.language_encoder.set_dataset_id(self.current_task)
        if self.current_task > 0:
            self.freeze_model(self.encoder)
            self.verify_frozen_state(self.encoder)
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
 
    def generate_pseudo_samples(self, current_features):
        pseudo_features = []
        pseudo_actions = []
        
        all_features = []
        all_actions = []
        for task_id in range(self.current_task):
            all_features.append(self.selected_features_per_task[task_id])
            all_actions.append(self.selected_actions_per_task[task_id])
        

        if all_features:
            all_features = torch.cat(all_features, dim=0) 
            all_actions = torch.cat(all_actions, dim=0)

            if not hasattr(self, 'used_indices'):
                self.used_indices = set()
                
            if self.count == 0:
                self.used_indices = set()
                
            total_samples = len(all_features)
            all_possible_indices = set(range(total_samples))
            available_indices = list(all_possible_indices - self.used_indices)
            
            if len(available_indices) >= self.sample_size_per_task:
                sample_size = min(self.sample_size_per_task, len(available_indices))
                selected_indices = torch.tensor(random.sample(available_indices, sample_size))
            else:
                self.used_indices = set()
                sample_size = self.sample_size
                selected_indices = torch.tensor(random.sample(all_possible_indices , sample_size))
                
            self.used_indices.update(selected_indices.tolist())
            
            pseudo_features = all_features[selected_indices]
            pseudo_actions = all_actions[selected_indices]
            
        return pseudo_features, pseudo_actions


    def observe(self, data):

        data = self.map_tensor_to_device(data)
        # with torch.no_grad():
        current_features = self.forward(data)

        self.optimizer.zero_grad()

        #genrate_data
        if self.current_task > 0:            
            # with torch.no_grad():
            past_feature, past_action = self.generate_pseudo_samples(current_features)      
            data['feature'] = torch.cat([current_features, past_feature], dim=0)
            data['actions'] = torch.cat([data['actions'], past_action], dim=0)
        else:
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

    def reset_losses(self, Epoch, idx, total_batches):
        print(
            f"[batch Info] Epoch: {Epoch:3d} | batch_idx: {idx:3d} | "
            f"model loss: {self.model_loss :5.6f} | "
        )

    def end_task(self, dataset, task_id, benchmark, env=None):

        if self.use_cluster_sample:

            dataset_size = len(dataset)
            batch_size = dataset_size // 16
            all_features = []
            all_actions = []

            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                indices = range(start_idx, end_idx)

                data_list = [dataset[i] for i in indices]
                data = self.merge_data(data_list)
                data = self.map_tensor_to_device(data)
                
                with torch.no_grad():
                    batch_features = self.forward(data)
                    torch.cuda.empty_cache()
                    batch_actions = data['actions']
                    all_features.append(batch_features)
                    all_actions.append(batch_actions)
                del data_list
                del data

            features = torch.cat(all_features, dim=0)
            actions = torch.cat(all_actions, dim=0)

            self.selected_features_per_task[task_id], self.selected_actions_per_task[task_id] = self.kmeans_sampling(
                features,
                actions,
                n_samples=self.sample_size,
                seed=self.cfg.seed
            )
        else:
            if self.use_random_sample:
                np.random.seed(self.cfg.seed)
                indices = np.random.choice(len(dataset), size=self.sample_size, replace=False)
            else:
                indices = range(self.sample_size)
            
            data_list = [dataset[i] for i in indices]
            data = self.merge_data(data_list)
            data = self.map_tensor_to_device(data)     

            with torch.no_grad():
                self.selected_features_per_task[task_id] = self.forward(data)
                torch.cuda.empty_cache()
                self.selected_actions_per_task[task_id] = data['actions']

    def merge_data(self, data_list):
        n = len(data_list)  
        
        merged_data = {}
        
        actions_list = [torch.from_numpy(d['actions']) for d in data_list]
        merged_data['actions'] = torch.stack(actions_list, dim=0)  # 1000x16x7
        
        merged_data['obs'] = {}
        obs_keys = data_list[0]['obs'].keys()
        for key in obs_keys:
            obs_list = [torch.from_numpy(d['obs'][key]) for d in data_list]
            merged_data['obs'][key] = torch.stack(obs_list, dim=0)
        
        task_emb = data_list[0]['task_emb'] 
        merged_data['task_emb'] = task_emb.repeat(n, 1)  
        
        return merged_data
    
    def kmeans_sampling(self, features, actions, n_samples, seed=None):

        features_np = features.cpu().numpy()
        
        # K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_samples, random_state=seed)
        kmeans.fit(features_np)
        
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features_np - center, axis=1)
            closest_point_idx = np.argmin(distances)
            selected_indices.append(closest_point_idx)
        
        selected_features = features[selected_indices]
        selected_actions = actions[selected_indices]
        
        return selected_features, selected_actions
    
