import torch
import torch.nn as nn
import torch.nn.functional as F
class ResGCONV(nn.Module):
    def __init__(self, input_dim):  
        super().__init__()
        self.U = nn.Parameter(torch.eye(input_dim)) 
        self.W = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, X, A_hat):
        XU = torch.matmul(X, self.U)  # [batch_size, n, input_dim]
        XW = torch.matmul(X, self.W)  # [batch_size, n, input_dim]
        AXW = torch.bmm(A_hat, XW)    # [batch_size, n, input_dim]
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
        batch_size = b.size(0)
        
        outputs = []
        for i in range(batch_size):
            n = original_sizes[i]
            
            b_i = b[i, :n].unsqueeze(0)  # [1, n]
            A_hat_i = A_hat[i, :n, :n].unsqueeze(0)  # [1, n, n]
            
            tau = torch.norm(b_i, dim=1, keepdim=True) + 1e-8
            b_scaled = b_i * (torch.sqrt(torch.tensor(n, device=self.device)) / tau)
            
            X = self.encoder(b_scaled.unsqueeze(-1))  # [1, n, gconv_dim]
            
            for layer in self.layers:
                X = layer(X, A_hat_i)
            
            decoded = self.decoder(X).squeeze(-1)  # [1, n]
            result = decoded * (tau / torch.sqrt(torch.tensor(n, device=self.device)))
            outputs.append(result.squeeze(0))  # [n]
        
        return outputs

