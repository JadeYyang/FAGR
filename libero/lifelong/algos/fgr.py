import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.fftpack import dct
from libero.lifelong.algos.base import Sequential
from r3m import load_r3m  # type: ignore
import numpy as np
from libero.lifelong.models import *
import random
from sklearn.cluster import KMeans
import pickle
from sklearn.covariance import ledoit_wolf
class FGR(Sequential):
    '''
    Feature generation replay(without filter)
    '''
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.sample_size = self.cfg.lifelong.sample_size  # Number of cluster center(n)

        # Load the encoder
        self.encoder = load_r3m(self.cfg.model_name)
        self.encoder.train() 
        self.encoder.to(cfg.device)

        self.features_center_per_task = [[] for _ in range(n_tasks)]
        self.features_cov_per_task = [[] for _ in range(n_tasks)]
        self.lang_embd_per_task = [[] for _ in range(n_tasks)]
        # Transformations for data augmentation (if needed)
        self.transform_size = transforms.CenterCrop(224)#
        self.feature_dim = self.cfg.lifelong.extra_dim * cfg.policy.n_obs_steps + self.cfg.lifelong.encoder_dim
        self.languge_dim = cfg.policy.text_embed_size
        self.model_loss = 0
        self.old_policy = None
        self.samples_per_cluster = self.cfg.lifelong.samples_per_cluster
        self.shrinkage_factor = self.cfg.lifelong.shrinkage_factor
        self.save_cluster = self.cfg.lifelong.save_cluster
        self.save_path = self.cfg.lifelong.save_path
        self.past_batch_size = self.cfg.lifelong.past_batch_size

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
        # r_z = F.normalize(r_z, p=2, dim=1)
        
        return r_z

    def get_extra_lang(self, data):
        extra_lang = self.policy.object_encoder(data)
        return extra_lang
 
    def generate_pseudo_samples(self):  
        
        # contatct all past data into a buffer
        all_centers = []
        all_covs = []
        for task_id in range(self.current_task):
            all_centers.append(self.features_center_per_task[task_id])
            all_covs.append(self.features_cov_per_task[task_id])
        
       
        if all_centers:
            all_centers = torch.cat(all_centers, dim=0)  # (self.sample_size * self.current_task, ...)
            all_covs = torch.cat(all_covs, dim=0)
            generated_features = []
            task_total = len(all_centers)
            for cluster_idx in range(task_total):
                
                mu = all_centers[cluster_idx]
                cov = all_covs[cluster_idx]
                L = torch.linalg.cholesky(cov)
                
                for _ in range(self.samples_per_cluster):
                    z = torch.randn_like(mu, device=self.cfg.device)
                    x = mu + L @ (z * self.shrinkage_factor)
                    
                    task_id = cluster_idx // self.sample_size
                    lang_embd = self.lang_embd_per_task[task_id]
                    combined_feature = torch.cat([x, lang_embd], dim=-1)
                    generated_features.append(combined_feature)

        generated_features = torch.stack(generated_features)
        cond_data = torch.zeros(
            size=(generated_features.shape[0], self.policy.horizon, self.policy.action_dim), 
            device=self.cfg.device, 
            dtype=generated_features.dtype
        )     
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)  
        generated_actions = self.old_policy.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=generated_features
        )            
        return generated_features, generated_actions

    def sample_past_data(self):

        if self.count == 0:
            if hasattr(self, 'past_features_pool'):
                del self.past_features_pool
            if hasattr(self, 'past_actions_pool'):
                del self.past_actions_pool
            if hasattr(self, 'sampling_indices'):
                del self.sampling_indices

        if (not hasattr(self, 'past_features_pool') or 
            not hasattr(self, 'sampling_indices') or 
            len(set(range(self.past_features_pool.shape[0])) - self.sampling_indices) < self.past_batch_size):
            
            with torch.no_grad():
                self.past_features_pool, self.past_actions_pool = self.generate_pseudo_samples()
                self.sampling_indices = set()

        total_samples = self.past_features_pool.shape[0]
        available_indices = list(set(range(total_samples)) - self.sampling_indices)
        selected_indices = torch.tensor(random.sample(available_indices, self.past_batch_size))
        self.sampling_indices.update(selected_indices.tolist())
        
        return self.past_features_pool[selected_indices], self.past_actions_pool[selected_indices]

    def observe(self, data):

        data = self.map_tensor_to_device(data)
        # with torch.no_grad():
        current_features = self.forward(data)

        self.optimizer.zero_grad()

        #genrate_data
        if self.current_task > 0:            
            sampled_features, sampled_actions = self.sample_past_data()      
            data['feature'] = torch.cat([current_features, sampled_features], dim=0)
            data['actions'] = torch.cat([data['actions'], sampled_actions], dim=0)
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

        dataset_size = len(dataset)
        batch_size = dataset_size // 16
        all_features = []

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            indices = range(start_idx, end_idx)

            data_list = [dataset[i] for i in indices]
            data = self.merge_data(data_list)
            data = self.map_tensor_to_device(data)
            
            with torch.no_grad():
                batch_features = self.forward(data)
                torch.cuda.empty_cache()
                all_features.append(batch_features)
            del data_list
            del data

        features = torch.cat(all_features, dim=0)[:, : self.feature_dim]
        self.lang_embd_per_task[task_id] = batch_features[0, self.feature_dim : ]

        self.features_center_per_task[task_id], self.features_cov_per_task[task_id] = self.kmeans_sampling(
            features,
            n_samples=self.sample_size,
            seed=self.cfg.seed
        )
        self.old_policy = get_policy_class(self.cfg.policy.policy_type)(self.cfg, self.cfg.shape_meta, n_task=self.n_tasks)
        self.old_policy.to(self.cfg.device)
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        

        for param in self.old_policy.parameters():
            param.requires_grad = False  

    def merge_data(self, data_list):

        n = len(data_list)  
        
        merged_data = {}
        
        actions_list = [torch.from_numpy(d['actions']) for d in data_list]
        merged_data['actions'] = torch.stack(actions_list, dim=0)  
        
        merged_data['obs'] = {}
        obs_keys = data_list[0]['obs'].keys()
        for key in obs_keys:
            obs_list = [torch.from_numpy(d['obs'][key]) for d in data_list]
            merged_data['obs'][key] = torch.stack(obs_list, dim=0)
        

        task_emb = data_list[0]['task_emb'] 
        merged_data['task_emb'] = task_emb.repeat(n, 1)  
        
        return merged_data
    
    def kmeans_sampling(self, features, n_samples, seed=None):
        """
        Args:
            features: torch.Tensor
            n_samples: int
            seed: int
        
        Returns:
            selected_features: torch.Tensor
            selected_actions: torch.Tensor
        """
        def convert_to_torch(covariance_matrices, device, dtype=torch.float32):
            # 转换为tensor
            cov_torch = torch.from_numpy(covariance_matrices).to(device, dtype=dtype)
            for cov in covariance_matrices:
                if not np.all(np.linalg.eigvals(cov) > 0):
                    print(f"Warning: Non-positive definite matrix detected before conversion")            
            
            diag_add = torch.eye(cov_torch.shape[-1], device=device, dtype=dtype) * 1e-4
            cov_torch = cov_torch + diag_add
            
            return cov_torch

        if self.current_task == 5:
            print("asd")
        features_np = features.cpu().numpy()

        kmeans = KMeans(n_clusters=n_samples, random_state=seed)
        kmeans.fit(features_np)
        cluster_labels = kmeans.fit_predict(features_np)

        cluster_centers = kmeans.cluster_centers_
        data_to_save = {
            'features': features_np,
            'labels': cluster_labels,
            'kmeans_model': kmeans
        }        
        if self.save_cluster:
            import os

            os.makedirs(self.save_path, exist_ok=True)
            
            save_path = os.path.join(self.save_path, f'clustering_data_{self.current_task}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)

        covariance_matrices = []

        for i in range(n_samples):
            cluster_mask = (cluster_labels == i)
            cluster_points = features_np[cluster_mask]          
            if len(cluster_points) > 1:

                cov_matrix, _ = ledoit_wolf(cluster_points)

                cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
            else:

                cov_matrix = np.eye(cluster_points.shape[1]) * 1e-6
                
            covariance_matrices.append(cov_matrix)
        
        cluster_centers = torch.from_numpy(cluster_centers).to(self.cfg.device, dtype=torch.float32)
        for cov in covariance_matrices:
            if not np.all(np.linalg.eigvals(cov) > 0):
                print(f"Warning: Non-positive definite matrix detected before conversion")

        covariance_matrices = np.array(covariance_matrices)
        covariance_matrices = convert_to_torch(covariance_matrices, self.cfg.device)

        return cluster_centers, covariance_matrices
    
