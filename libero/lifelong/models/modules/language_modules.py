"""
This file contains modules that encode language task embeddings.
"""
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """
    Dummy encoder that directly outputs the pretrained task embedding
    """

    def __init__(self, dummy=True):
        super().__init__()

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = data["task_emb"]  # (B, L, H)
        return h


class MLPEncoder(nn.Module):
    """
    Encode task embedding
    h = f(e), where
        e: pretrained task embedding from large model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, num_encoders=None):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        
        if num_encoders == 'None':
            # 单个MLP的情况
            layers = []
            for i in range(num_layers - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
            self.projection = nn.Sequential(*layers)
            self.multi_encoder = False
        else:
            # 多个MLP的情况
            self.projections = nn.ModuleList()
            for _ in range(num_encoders):
                layers = []
                for i in range(num_layers - 1):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1]))
                self.projections.append(nn.Sequential(*layers))
            self.multi_encoder = True
            self.current_dataset_id = None

    def set_dataset_id(self, dataset_id):
        """设置当前训练的数据集ID，并相应地冻结/解冻参数"""
        if not self.multi_encoder:
            return
        self.current_dataset_id = dataset_id
        # 冻结所有projection
        for i, projection in enumerate(self.projections):
            for param in projection.parameters():
                if i == dataset_id:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        if not self.multi_encoder:
            return self.projection(data["task_emb"])
            
        assert self.current_dataset_id is not None, "Please set dataset_id first"
        h = self.projections[self.current_dataset_id](data["task_emb"])
        return h

    # def forward(self, data):
    #     """
    #     data:
    #         task_emb: (B, E)
    #     """
    #     h = self.projection(data["task_emb"])  # (B, H)
    #     return h
