import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from gnp_model import GNP
from visualize import plot_loss, plot_spectral_radius, show_progress, plot_condition_numbers, plot_learning_rate
from solver_comparison import run_gmres
from data_generator import generate_test_matrices, normalize_matrix, generate_fourth_order_pv_matrices, generate_fem_test_matrices, generate_fem_samples, make_samples_from_dense_mats, generate_strict_diagonally_dominant, generate_clean_fourth_order_matrix, generate_clean_fourth_order_pv_matrix, generate_fem_triangle_matrix, generate_fem_circle_matrix

def split_samples(samples, ratio):
    samples_copy = samples[:]
    random.shuffle(samples_copy)
    split_idx = int(len(samples_copy) * (1 - ratio))
    return samples_copy[:split_idx], samples_copy[split_idx:]

class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def dynamic_collate(batch):
    b_list, x_list, K_list, K_norm_list = zip(*batch)
    
    max_n = max(b.size(0) for b in b_list)
    
    b_tensor = []
    x_tensor = []
    K_tensor = []
    K_norm_tensor = []
    original_sizes = []
    
    for b, x, K, K_norm in zip(b_list, x_list, K_list, K_norm_list):
        n = b.size(0)
        original_sizes.append(n)
        
        b_padded = torch.zeros(max_n, dtype=b.dtype)
        b_padded[:n] = b
        b_tensor.append(b_padded)
        
        x_padded = torch.zeros(max_n, dtype=x.dtype)
        x_padded[:n] = x
        x_tensor.append(x_padded)
        
        K_padded = torch.zeros((max_n, max_n), dtype=K.dtype)
        K_padded[:n, :n] = K
        K_tensor.append(K_padded)
        
        K_norm_padded = torch.zeros((max_n, max_n), dtype=K_norm.dtype)
        K_norm_padded[:n, :n] = K_norm
        K_norm_tensor.append(K_norm_padded)
    
    b_tensor = torch.stack(b_tensor, dim=0)  
    x_tensor = torch.stack(x_tensor, dim=0)  
    K_tensor = torch.stack(K_tensor, dim=0)  
    K_norm_tensor = torch.stack(K_norm_tensor, dim=0)  
    
    return b_tensor, x_tensor, K_tensor, K_norm_tensor, original_sizes

def estimate_inverse(model, A, device='cpu'):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    n = A.shape[-1] 
    I = torch.eye(n).to(device).float()
    A_tensor = A.unsqueeze(0) if A.dim() == 2 else A 
    
    estimated_inverse = []
    for i in range(n):
        e_i = I[i].unsqueeze(0)  # [1, n]
        with torch.no_grad():
            x_hat_list = model(e_i, A_tensor, [n]) 
            x_hat = x_hat_list[0] 
        
        estimated_inverse.append(x_hat.squeeze().cpu().numpy())  
    
    return np.stack(estimated_inverse, axis=1)

def calculate_spectral(model, reference_matrix, device):
    if isinstance(reference_matrix, torch.Tensor):
        reference_matrix = reference_matrix.cpu().numpy()
    
    A_norm = normalize_matrix(reference_matrix) 
    A_norm_tensor = torch.tensor(A_norm, dtype=torch.float32, device=device)
    if A_norm_tensor.dim() == 2:
        A_norm_tensor = A_norm_tensor.unsqueeze(0)  
    
    A_inv_estimated = estimate_inverse(model, A_norm_tensor, device)
    A_inv_estimated = torch.tensor(A_inv_estimated, dtype=torch.float32).to(device)
    A = torch.tensor(reference_matrix, dtype=torch.float32).to(device)
    A_inv_A = A_inv_estimated @ A
    diff_matrix = torch.eye(A.size(0)).to(device).float() - A_inv_A
    
    eigenvalues, _ = torch.linalg.eig(diff_matrix.cpu())
    max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
    min_eigenvalue = torch.min(torch.abs(eigenvalues)).item()
    return {'max': max_eigenvalue, 'min': min_eigenvalue}

def create_M_operator(model, A_hat, device):
    def M_operator(v):
        v_tensor = torch.tensor(v, dtype=torch.float32).view(1, -1).to(device)
        A_hat_tensor = torch.tensor(A_hat, dtype=torch.float32).unsqueeze(0).to(device)
        n = v_tensor.size(1)
        with torch.no_grad():
            output_list = model(v_tensor, A_hat_tensor, [n])
        output_tensor = output_list[0]
        return output_tensor.squeeze(0).detach().cpu().numpy()
    return M_operator

def train_and_validate(model, optimizer, scheduler, epochs, train_loader, val_loader, device, save_dir, total_epochs, tol):
    model.train()
    train_losses, val_losses = [], []
    spectral_data = {'max': [], 'min': []}  
    prev_train_loss = float('inf')  
    reference_matrix = train_loader.dataset.samples[0][2].numpy()
    lr_history = []  
    
    # L1正则化系数
    l1_lambda = 1e-3 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            b_tensor, x_tensor, _, K_norm_tensor, original_sizes = batch
            
            b_tensor = b_tensor.to(device)
            K_norm_tensor = K_norm_tensor.to(device)
            x_tensor = x_tensor.to(device) 
            
            output = model(b_tensor, K_norm_tensor, original_sizes)
            
            mse_loss = 0
            for i, n in enumerate(original_sizes):
                mse_loss += F.l1_loss(output[i][:n], x_tensor[i, :n])

            l1_reg = torch.tensor(0., requires_grad=True).to(device)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
            
            loss = mse_loss + l1_lambda * l1_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
            lr_history.append(optimizer.param_groups[0]['lr'])
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                b_val, x_val, _, K_norm_val, original_sizes_val = val_batch
                
                b_val = b_val.to(device)
                K_norm_val = K_norm_val.to(device)
                x_val = x_val.to(device) 
                
                outputs = model(b_val, K_norm_val, original_sizes_val)
                
                for i, n in enumerate(original_sizes_val):
                    total_val_loss += F.l1_loss(outputs[i][:n], x_val[i, :n]).item()
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        loss_delta = abs(avg_train_loss - prev_train_loss)
        if loss_delta < tol:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        prev_train_loss = avg_train_loss
        
        spectral = calculate_spectral(model, reference_matrix, device)
        spectral_data['max'].append(spectral['max'])
        spectral_data['min'].append(spectral['min'])
        
        show_progress(epoch + 1, total_epochs)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    plot_loss(train_losses, val_losses, save_dir=save_dir)
    plot_spectral_radius(spectral_data, save_dir=save_dir)
    plot_learning_rate(lr_history, save_dir=save_dir)
    
    return model

def main():
    mlp_hidden = 16         
    gconv_dim = 32
    num_layers = 8           
    epochs = 800             
    batch_size = 16          
    lr = 1e-3               
    save_dir = './figures'   
    tol = 1e-6               
    test_n = 500            
    test_h = 2e-0           
    test_iter = 10000      
    val_ratio = 0.1  

    warmup_ratio = 0.05  
    stable_ratio = 0.15  
    decay_ratio = 0.1   
    decay_rate = 0.5     
    min_lr = 5e-5      

    # 1. 有限差分矩阵
    fd_mats = generate_test_matrices(
        n=500, num=30, h_range=[1e-1, 1e-0], only_fourth_order=True
    )
    train_samples_fd_full = make_samples_from_dense_mats(
        fd_mats, num_samples_per_matrix=10
    )
    
    # 2. 混合矩阵
    mix_mats = generate_test_matrices(
        n=500, num=30, h_range=[1e-1, 1e-0], only_fourth_order=False
    )
    train_samples_mix_full = make_samples_from_dense_mats(
        mix_mats, num_samples_per_matrix=10
    )
    
    # 3. FEM矩阵
    train_samples_fem_full = generate_fem_samples(
        num_matrices=60, tri_ratio=0.5, tri_nx=33,
        circ_nr=40, circ_ntheta0=25, num_samples_per_matrix=10, pv_coeff=1.0
    )
    
    pv_mats = generate_fourth_order_pv_matrices(
        n=500, num=30, h_range=[1e-1, 1e-0], pv_coeff=1.0
    )
    train_samples_pv_full = make_samples_from_dense_mats(
        pv_mats, num_samples_per_matrix=10
    )

    train_fd, val_fd = split_samples(train_samples_fd_full, val_ratio)
    train_mix, val_mix = split_samples(train_samples_mix_full, val_ratio)
    train_fem, val_fem = split_samples(train_samples_fem_full, val_ratio)
    train_pv, val_pv = split_samples(train_samples_pv_full, val_ratio)

    train_samples = train_fd + train_mix + train_fem + train_pv
    val_samples = val_fd + val_mix + val_fem + val_pv

    train_dataset = CustomDataset(samples=train_samples)
    val_dataset = CustomDataset(samples=val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dynamic_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GNP(
        mlp_hidden=mlp_hidden, gconv_dim=gconv_dim, num_layers=num_layers, device=device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    warmup_epochs = int(epochs * warmup_ratio)
    stable_epochs = int(epochs * stable_ratio)
    decay_epochs = int(epochs * decay_ratio)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        elif epoch < warmup_epochs + stable_epochs:
            return 1.0
        else:
            step = (epoch - warmup_epochs - stable_epochs) // decay_epochs
            calculated_lr = lr * (decay_rate ** step)
            return max(calculated_lr / lr, min_lr / lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=-1
    )

    trained_model = train_and_validate(
        model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
        train_loader=train_loader, val_loader=val_loader, device=device,
        save_dir=save_dir, total_epochs=epochs, tol=tol
    )

    fd_mat = generate_test_matrices(
        n=test_n, num=40, h_range=[1e-1, 1e-0], only_fourth_order=True
    )
    
    mix_mat = generate_test_matrices(
        n=test_n, num=40, h_range=[1e-1, 1e-0], only_fourth_order=False
    )
    test_matrices_pv = generate_fourth_order_pv_matrices(
        n=test_n, num=40, h_range=[1e-1, 1e-0], pv_coeff=1.0
    )
    
    test_matrices_fem = generate_fem_test_matrices(
        num=80, tri_ratio=0.5, tri_nx=33, circ_nr=40, circ_ntheta0=25, pv_coeff=1.0
    )
    
    test_matrices = fd_mat + mix_mat + test_matrices_pv + test_matrices_fem
    
    original_cond = []
    preconditioned_cond = []
    valid_count = 0
    
    trained_model.eval()
    with torch.no_grad():
        for A_raw in test_matrices:
            try:
                A_norm = normalize_matrix(A_raw)
                A_inv = estimate_inverse(trained_model, A_norm, device)
                product = A_inv @ A_raw 
                if np.any(np.isnan(product)) or np.any(np.isinf(product)):
                    continue
                cond_orig = np.linalg.cond(A_raw)
                cond_pre = np.linalg.cond(product)
                if np.isfinite(cond_orig) and np.isfinite(cond_pre):
                    original_cond.append(cond_orig)
                    preconditioned_cond.append(cond_pre)
                    valid_count += 1
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    print(f"\nValid results: {valid_count}/{len(test_matrices)}")
    if valid_count > 0:
        plot_condition_numbers(original_cond, preconditioned_cond, save_dir=save_dir)
    else:
        print("No valid data for plotting")

    # 生成并测试五种不同类型的矩阵
    matrix_types = [
        "强对角占优矩阵",
        "纯净的四阶差分矩阵", 
        "有PV项的四阶差分矩阵",
        "有PV项的三角形FEM矩阵",
        "有PV项的圆形FEM矩阵"
    ]

    pv_coeff = 1.0 
    tri_nx = 33    
    circ_nr = 40   
    circ_ntheta0 = 25

    for i, matrix_type in enumerate(matrix_types):
        
        if matrix_type == "强对角占优矩阵":
            A_test = generate_strict_diagonally_dominant(test_n)
            x_true = np.random.randn(test_n).astype(np.float32)
            b_test = A_test @ x_true
            
        elif matrix_type == "纯净的四阶差分矩阵":
            A_test = generate_clean_fourth_order_matrix(test_n, test_h)
            x_true = np.random.randn(test_n).astype(np.float32)
            b_test = A_test @ x_true
            
        elif matrix_type == "有PV项的四阶差分矩阵":
            A_test = generate_clean_fourth_order_pv_matrix(test_n, test_h, pv_coeff)
            x_true = np.random.randn(test_n).astype(np.float32)
            b_test = A_test @ x_true
            
        elif matrix_type == "有PV项的三角形FEM矩阵":
            A_test = generate_fem_triangle_matrix(nx=tri_nx, pv_coeff=pv_coeff)
            x_true = np.random.randn(A_test.shape[0]).astype(np.float32)
            b_test = A_test @ x_true
            
        elif matrix_type == "有PV项的圆形FEM矩阵":
            A_test = generate_fem_circle_matrix(nr=circ_nr, ntheta0=circ_ntheta0, pv_coeff=pv_coeff)
            x_true = np.random.randn(A_test.shape[0]).astype(np.float32)
            b_test = A_test @ x_true
        
        A_hat = normalize_matrix(A_test)
        M_operator = create_M_operator(trained_model, A_hat, device)
        
        # 运行GMRES和FGMRES
        x_gmres, iters_gmres = run_gmres(
            A_test, b_test, M=None, tol=1e-4, m=50, maxiter=test_iter
        )
        x_fgmres, iters_fgmres = run_gmres(
            A_test, b_test, M=M_operator, tol=1e-4, m=50, maxiter=test_iter
        )
        
        res_gmres = np.linalg.norm(A_test @ x_gmres - b_test)
        res_fgmres = np.linalg.norm(A_test @ x_fgmres - b_test)
        print(f"{matrix_type} GMRES迭代次数: {iters_gmres}")
        print(f"{matrix_type} FGMRES迭代次数: {iters_fgmres}")
        print(f"{matrix_type} GMRES残差: {res_gmres:.4e}")
        print(f"{matrix_type} FGMRES残差: {res_fgmres:.4e}")


if __name__ == "__main__":
    main()

