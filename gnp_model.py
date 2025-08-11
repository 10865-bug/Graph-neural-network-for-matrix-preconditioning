import torch
import torch.nn as nn
import torch.nn.functional as F
class ResGCONV(nn.Module):
    def __init__(self, input_dim):  
        super().__init__()
        self.U = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.W)

    def forward(self, X, A_hat):
        batch_size, n, input_dim = X.shape

        # 计算 XU 和 XW
        XU = torch.matmul(X, self.U)  # [batch_size, n, input_dim]
        XW = torch.matmul(X, self.W)  # [batch_size, n, input_dim]
        
        # 批量矩阵乘法 AXW
        AXW = torch.bmm(A_hat, XW)    # [batch_size, n, input_dim]
        
        # 返回激活后的结果
        return F.relu(XU + AXW)
    
class GNP(nn.Module):
    def __init__(self, mlp_hidden, gconv_dim, num_layers, device):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, gconv_dim),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([ResGCONV(gconv_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(gconv_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.device = device
        self.to(device)
        
    def forward(self, b, A_hat, original_sizes):
        """修改后的前向传播，支持不同尺寸的矩阵"""
        batch_size, max_n = b.size(0), A_hat.size(-1)
        
        outputs = []
        for i in range(batch_size):
            n = original_sizes[i]  # 每个样本的实际尺寸
            
            # 只处理实际尺寸部分
            b_i = b[i, :n].unsqueeze(0)  # [1, n]
            A_hat_i = A_hat[i, :n, :n].unsqueeze(0)  # [1, n, n]
            
            # 输入归一化（基于实际尺寸）
            tau = torch.norm(b_i, dim=1, keepdim=True) + 1e-8
            b_scaled = b_i * (torch.sqrt(torch.tensor(n, device=self.device)) / tau)
            
            # 编码器处理
            X = self.encoder(b_scaled.unsqueeze(-1))  # [1, n, gconv_dim]
            
            # 图卷积层（仅处理有效区域）
            for layer in self.layers:
                X = layer(X, A_hat_i)
            
            # 解码器输出
            decoded = self.decoder(X).squeeze(-1)  # [1, n]
            result = decoded * (tau / torch.sqrt(torch.tensor(n, device=self.device)))
            outputs.append(result.squeeze(0))  # [n]
        
        return outputs