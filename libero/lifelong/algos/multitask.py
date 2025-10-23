import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *
from torchvision import transforms
from r3m import load_r3m  

class Multitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.transform_size = transforms.CenterCrop(224)#
        # Load the encoder
        self.encoder = load_r3m(self.cfg.model_name)
        self.encoder.train() 
        self.encoder.to(cfg.device)

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
    
    def learn_all_tasks(self, datasets, benchmark, result_summary):
        self.start_task(-1)
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=False,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            if epoch > 0 or (self.cfg.pretrain):  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                t0 = time.time()
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"multitask_model_ep{epoch}.pth"
                )
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epochs on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)

                if prev_success_rate < success_rate and (not self.cfg.pretrain):
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                if self.cfg.lifelong.eval_in_train:
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                        + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                        flush=True,
                    )

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best policy if there is any
        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        self.end_task(concat_dataset, -1, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best_succ = losses[idx_at_best_succ]
            success_at_best_succ = successes[idx_at_best_succ]
            losses[idx_at_best_succ:] = loss_at_best_succ
            successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
