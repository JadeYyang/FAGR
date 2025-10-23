import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.fftpack import dct
from libero.lifelong.algos.base import Sequential
from libero.libero.utils.load_r3m import load_r3m  # type: ignore
import numpy as np
from libero.lifelong.models import *
import random
from sklearn.cluster import KMeans
import pickle
from sklearn.covariance import ledoit_wolf
class FGRA(Sequential):
    '''
    Feature-Action Generation Replay
    '''
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.sample_size = self.cfg.lifelong.sample_size  # Number of cluster center(n)

        # Load the encoder
        self.encoder = load_r3m(self.cfg.r3m_folder, self.cfg.model_name, cfg.device)
        self.encoder.train() 

        self.features_center_per_task = [[] for _ in range(n_tasks)]
        self.features_cov_per_task = [[] for _ in range(n_tasks)]
        self.actions_center_per_task = [[] for _ in range(n_tasks)]
        self.actions_cov_per_task = [[] for _ in range(n_tasks)]
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
        self.confidence_threshold = self.cfg.lifelong.confidence_threshold
        self.action_step = cfg.policy.n_action_steps

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
 
    def generate_pseudo_samples(self):

        all_centers, all_covs, all_action_centers, all_action_covs = self._merge_task_data()
        if len(all_centers) == 0:
            return None, None
        
        final_features = []
        final_actions = []
        max_attempts = 3  
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1

            generated_features, cluster_indices = self._generate_features(
                all_centers, all_covs)
            

            filtered_features, filtered_actions, filtered_counts = self._generate_and_filter_actions(
                generated_features, cluster_indices, all_action_centers, all_action_covs)
            
            if len(filtered_features) > 0:
                final_features.append(filtered_features)
                final_actions.append(filtered_actions)
        
        if not final_features:
            return None, None
            
        return torch.cat(final_features, dim=0), torch.cat(final_actions, dim=0)

    def _generate_features(self, all_centers, all_covs):
        generated_features = []
        cluster_indices = []
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
                cluster_indices.append(cluster_idx)
                    
        generated_features = torch.stack(generated_features)
        cluster_indices = torch.tensor(cluster_indices, device=self.cfg.device)
        
        return generated_features, cluster_indices

    def _generate_and_filter_actions(self, generated_features, cluster_indices, 
                                all_action_centers, all_action_covs):
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

        generated_actions_filtered = torch.mean(generated_actions[:, :self.action_step, :], dim=1)        

        filtered_features = []
        filtered_actions = []
        filtered_indices = []
        filtered_counts = {}  
        task_total = len(all_action_centers)
        
        for cluster_idx in range(task_total):
            cluster_mask = (cluster_indices == cluster_idx)
            cluster_features = generated_features[cluster_mask]
            cluster_actions = generated_actions[cluster_mask]
            cluster_actions_filtered = generated_actions_filtered[cluster_mask]

            if len(cluster_features) == 0:
                filtered_counts[cluster_idx] = 0
                continue

            action_center = all_action_centers[cluster_idx]
            action_cov = all_action_covs[cluster_idx]
            
            diff = (cluster_actions_filtered - action_center)
            inv_cov = torch.linalg.inv(action_cov)
            mahalanobis_dist = torch.sqrt(torch.sum(
                torch.matmul(diff, inv_cov) * diff, dim=1
            ))
            
            confident_mask = mahalanobis_dist < self.confidence_threshold
            
            filtered_features.extend(cluster_features[confident_mask])
            filtered_actions.extend(cluster_actions[confident_mask])
            filtered_indices.extend([cluster_idx] * confident_mask.sum().item())
            filtered_counts[cluster_idx] = confident_mask.sum().item()
        
        if not filtered_features:
            return torch.empty(0, device=self.cfg.device), torch.empty(0, device=self.cfg.device), {}
        
        filtered_counts['indices'] = torch.tensor(filtered_indices, device=self.cfg.device)
        return torch.stack(filtered_features), torch.stack(filtered_actions), filtered_counts
    
    def _merge_task_data(self):
        all_centers = []
        all_covs = []
        all_action_centers = []
        all_action_covs = []
        
        for task_id in range(self.current_task):
            all_centers.append(self.features_center_per_task[task_id])
            all_covs.append(self.features_cov_per_task[task_id])
            all_action_centers.append(self.actions_center_per_task[task_id])
            all_action_covs.append(self.actions_cov_per_task[task_id])
            
        if len(all_centers):
            all_centers = torch.cat(all_centers, dim=0)
            all_covs = torch.cat(all_covs, dim=0)
            all_action_centers = torch.cat(all_action_centers, dim=0)
            all_action_covs = torch.cat(all_action_covs, dim=0)
            
        return all_centers, all_covs, all_action_centers, all_action_covs


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
        loss = loss_model #+ self.inter_weight * inter_loss
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
                all_features.append(batch_features)
                all_actions.append(data['actions'])
            del data_list
            del data

        features = torch.cat(all_features, dim=0)[:, : self.feature_dim]
        actions = torch.cat(all_actions, dim=0)
        self.lang_embd_per_task[task_id] = batch_features[0, self.feature_dim : ]

        (
            self.features_center_per_task[task_id],
            self.actions_center_per_task[task_id],
            self.features_cov_per_task[task_id],
            self.actions_cov_per_task[task_id]
        ) = self.kmeans_sampling(
            features,
            actions,
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
    
    def kmeans_sampling(self, features, actions, n_samples, seed=None):
        def convert_to_torch(covariance_matrices, device, dtype=torch.float32):

            cov_torch = torch.from_numpy(covariance_matrices).to(device, dtype=dtype)
            for cov in covariance_matrices:
                if not np.all(np.linalg.eigvals(cov) > 0):
                    print(f"Warning: Non-positive definite matrix detected before conversion")            
            
            diag_add = torch.eye(cov_torch.shape[-1], device=device, dtype=dtype) * 1e-4
            cov_torch = cov_torch + diag_add
            
            return cov_torch
 
        features_np = features.cpu().numpy()
        actions_np = actions.cpu().numpy()

        actions_np = actions_np[:, :self.action_step , :]

        actions_mean = np.mean(actions_np, axis=1)  # n x 7        

        kmeans = KMeans(n_clusters=n_samples, random_state=seed)
        kmeans.fit(features_np)
        cluster_labels = kmeans.fit_predict(features_np)
        cluster_centers = kmeans.cluster_centers_
        data_to_save = {
            'features': features_np,
            'actions': actions_np,
            'labels': cluster_labels,
            'kmeans_model': kmeans
        }        
        if self.save_cluster:
            import os

            os.makedirs(self.save_path, exist_ok=True)
            
            save_path = os.path.join(self.save_path, f'clustering_data_{self.current_task}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)

        feature_covariance_matrices = []
        action_covariance_matrices = []
        action_centers = []

        for i in range(n_samples):
            cluster_mask = (cluster_labels == i)
            cluster_features = features_np[cluster_mask]
            cluster_actions = actions_mean[cluster_mask]
            if len(cluster_features) > 1:

                feature_cov, _ = ledoit_wolf(cluster_features)
                feature_cov += 1e-6 * np.eye(feature_cov.shape[0])
            else:
                feature_cov = np.eye(cluster_features.shape[1]) * 1e-6

            if len(cluster_actions) > 1:
                #Ledoit-Wolf
                action_cov, _ = ledoit_wolf(cluster_actions)

                action_cov += 1e-6 * np.eye(action_cov.shape[0])
                action_center = np.mean(cluster_actions, axis=0)
            else:
                action_cov = np.eye(7) * 1e-6  
                action_center = cluster_actions[0]
            
            feature_covariance_matrices.append(feature_cov)
            action_covariance_matrices.append(action_cov)
            action_centers.append(action_center)                

        cluster_centers = torch.from_numpy(cluster_centers).to(self.cfg.device, dtype=torch.float32)
        action_centers = torch.from_numpy(np.stack(action_centers)).to(self.cfg.device, dtype=torch.float32)
        for cov in feature_covariance_matrices:
            if not np.all(np.linalg.eigvals(cov) > 0):
                print(f"Warning: Non-positive definite matrix detected before conversion")
        feature_covariance_matrices = np.array(feature_covariance_matrices)
        feature_covariance_matrices = convert_to_torch(feature_covariance_matrices, self.cfg.device)
        action_covariance_matrices = torch.from_numpy(np.stack(action_covariance_matrices)).to(self.cfg.device, dtype=torch.float32)

        return cluster_centers, action_centers, feature_covariance_matrices, action_covariance_matrices

    
