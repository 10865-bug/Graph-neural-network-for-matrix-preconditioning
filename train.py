import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from gnp_model import GNP
from visualize import plot_loss, plot_spectral_radius, show_progress, plot_condition_numbers
from data_generator import generate_data, generate_test_matrices, normalize_matrix, generate_test_data, load_stand_train_data, load_stand_test_data  
from solver_comparison import run_gmres

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
    n = A.shape[-1]  # 实际矩阵尺寸
    I = torch.eye(n).to(device).float()
    A_tensor = A.unsqueeze(0) if A.dim() == 2 else A  # 确保形状为 [1, n, n]
    
    estimated_inverse = []
    for i in range(n):
        e_i = I[i].unsqueeze(0)  # [1, n]
        with torch.no_grad():
            x_hat_list = model(e_i, A_tensor, [n]) 
            x_hat = x_hat_list[0] 
        
        estimated_inverse.append(x_hat.squeeze().cpu().numpy())  # 现在 x_hat 是张量
    
    return np.stack(estimated_inverse, axis=1)

def calculate_spectral(model, reference_matrix, device):
    if isinstance(reference_matrix, torch.Tensor):
        reference_matrix = reference_matrix.cpu().numpy()
    
    A_norm = normalize_matrix(reference_matrix)  # 确保 A_norm 的形状为 [n, n]
    A_norm_tensor = torch.tensor(A_norm, dtype=torch.float32, device=device)
    if A_norm_tensor.dim() == 2:
        A_norm_tensor = A_norm_tensor.unsqueeze(0)  # 确保 A_norm_tensor 的形状为 [1, n, n]
    
    A_inv_estimated = estimate_inverse(model, A_norm_tensor, device)
    A_inv_estimated = torch.tensor(A_inv_estimated, dtype=torch.float32).to(device)
    A = torch.tensor(reference_matrix, dtype=torch.float32).to(device)
    A_inv_A = A_inv_estimated @ A
    diff_matrix = torch.eye(A.size(0)).to(device).float() - A_inv_A
    
    eigenvalues, _ = torch.linalg.eig(diff_matrix.cpu())
    max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
    min_eigenvalue = torch.min(torch.abs(eigenvalues)).item()
    return {'max': max_eigenvalue, 'min': min_eigenvalue}

def train_and_validate(model, optimizer, epochs, train_loader, val_loader, device, save_dir, total_epochs, tol):
    model.train()
    train_losses, val_losses = [], []
    spectral_data = {'max': [], 'min': []}  
    prev_train_loss = float('inf')  
    reference_matrix = train_loader.dataset.samples[0][2].numpy()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            b_tensor, x_tensor, K_tensor, K_norm_tensor, original_sizes = batch
            
            b_tensor = b_tensor.to(device)
            K_norm_tensor = K_norm_tensor.to(device)
            x_tensor = x_tensor.to(device)  # 提前移动目标张量到设备
            
            # 调用模型时传递 original_sizes
            output = model(b_tensor, K_norm_tensor, original_sizes)
            
            loss = 0
            for i, n in enumerate(original_sizes):
                # 确保只比较实际尺寸部分
                loss += F.l1_loss(output[i][:n], x_tensor[i, :n]) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                b_val, x_val, K_val, K_norm_val, original_sizes_val = val_batch
                
                b_val = b_val.to(device)
                K_norm_val = K_norm_val.to(device)
                x_val = x_val.to(device)  # 提前移动目标张量到设备
                
                # 调用模型时传递 original_sizes
                outputs = model(b_val, K_norm_val, original_sizes_val)
                
                for i, n in enumerate(original_sizes_val):
                    # 确保只比较实际尺寸部分
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
    
    return model

def main():
    mlp_hidden = 32         
    gconv_dim = 16         
    num_layers = 8           
    epochs = 100             
    batch_size = 16          
    lr = 1e-3               
    save_dir = './figures'   
    tol = 1e-4               
    test_n = 1000            
    test_h = 1e-0           
    test_iter = 10000      

    train_samples = load_stand_train_data(
        stand_train_dir="..\stand_small\stand_small_train",
        max_matrices=20,
        num_samples_per_matrix=20
    )

    val_samples = load_stand_test_data(
        stand_test_dir="..\stand_small\stand_small_test",
        max_matrices=10,
        num_samples_per_matrix=1
    )

    train_dataset = CustomDataset(samples=train_samples)
    val_dataset = CustomDataset(samples=val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  
        shuffle=True,
        collate_fn=dynamic_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  
        shuffle=False,
        collate_fn=dynamic_collate
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GNP(
        mlp_hidden=mlp_hidden, 
        gconv_dim=gconv_dim,
        num_layers=num_layers,
        device=device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trained_model = train_and_validate(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        total_epochs=epochs,
        tol=tol
    )

    test_matrices = generate_test_matrices(
        n=test_n, 
        num=100, 
        h_range=[1e-3, 1e-1],
        only_fourth_order=True
    )
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
        
    A_test_raw, b_test = generate_test_data(test_n, test_h)
    A_hat = normalize_matrix(A_test_raw)  

    def create_M_operator(model, A_hat, device):
        def M_operator(v):
            v_tensor = torch.tensor(v, dtype=torch.float32).view(1, -1).to(device)
            A_hat_tensor = torch.tensor(A_hat, dtype=torch.float32).unsqueeze(0).to(device)
            n = v_tensor.size(1)
            with torch.no_grad():  # 确保无梯度计算
                output_list = model(v_tensor, A_hat_tensor, [n])
            output_tensor = output_list[0]
            return output_tensor.squeeze(0).detach().cpu().numpy()  # 分离梯度
        return M_operator

    M_operator = create_M_operator(trained_model, A_hat, device) 

    x_gmres, iters_gmres = run_gmres(A_test_raw, b_test, M=None, tol=1e-4, m=50, maxiter=test_iter)
    x_fgmres, iters_fgmres = run_gmres(A_test_raw, b_test, M=M_operator, tol=1e-4, m=50, maxiter=test_iter)

    print(f"GMRES iterations: {iters_gmres}")
    print(f"FGMRES iterations: {iters_fgmres}")
    print(f"GMRES residual: {np.linalg.norm(A_test_raw @ x_gmres - b_test):.2e}")
    print(f"FGMRES residual: {np.linalg.norm(A_test_raw @ x_fgmres - b_test):.2e}")

if __name__ == "__main__":

    main()
