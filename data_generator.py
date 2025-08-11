import numpy as np
import scipy
import scipy.sparse as sp
import torch
import glob
import os

def generate_strict_diagonally_dominant(n):
    A = np.zeros((n, n))
    
    main_diag = np.random.uniform(20, 100, n)
    A[np.diag_indices(n)] = main_diag
    
    for offset in [-2, -1, 1, 2]:
        diag_len = n - abs(offset)
        off_diag = np.random.uniform(-10, 10, diag_len)
        np.fill_diagonal(A[offset:], off_diag)
    
    return A

def generate_clean_fourth_order_matrix(n, h):
    main = 6 / (h**4) * np.ones(n)
    off1 = -4 / (h**4) * np.ones(n-1)
    off2 = 1 / (h**4) * np.ones(n-2)
    
    diagonals = [off2, off1, main, off1, off2]
    offsets = [-2, -1, 0, 1, 2]
    A = sp.diags(diagonals, offsets, shape=(n, n)).toarray()
    return A.astype(np.float32)

def generate_test_matrices(n, num=100, h_range=[1e-2, 1e-1], only_pentadiagonal=False):
    test_mats = []
    
    if only_pentadiagonal:
        for _ in range(num):
            test_mats.append(generate_random_pentadiagonal_matrix(n))
    else:
        num_dominant = num // 2
        for _ in range(num_dominant):
            test_mats.append(generate_strict_diagonally_dominant(n))
        
        num_diff = num - num_dominant
        for _ in range(num_diff):
            h = np.random.uniform(h_range[0], h_range[1])
            test_mats.append(generate_clean_fourth_order_matrix(n, h))
    
    return test_mats

def generate_random_pentadiagonal_matrix(n,h):
    A = np.zeros((n, n))
    
    main_diag = np.random.uniform(20, 100, n)
    np.fill_diagonal(A, main_diag)
    
    for offset in [1, 2]:
        diag_len = n - offset
        off_diag = np.random.uniform(-10, 10, diag_len)
        np.fill_diagonal(A[offset:], off_diag)    
        np.fill_diagonal(A[:, offset:], off_diag)
    
    return A

def normalize_matrix(A):
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
    
    row_norm = np.max(np.sum(np.abs(A), axis=1))
    col_norm = np.max(np.sum(np.abs(A), axis=0))
    gamma = min(row_norm, col_norm) + 1e-12
    
    return A / gamma

def generate_data(num_matrices, n, num_samples_per_matrix=100, train_ratio=0.6, h_range=[1e-2, 5e-2]):
    all_matrices_raw = []  
    all_matrices_norm = [] 
    all_samples = []

    for matrix_idx in range(num_matrices):
        A_raw = generate_random_pentadiagonal_matrix(n, h_range)
        A_norm = normalize_matrix(A_raw)
        
        all_matrices_raw.append(A_raw)
        all_matrices_norm.append(A_norm)
        
        for _ in range(num_samples_per_matrix):
            x = np.random.randn(n).astype(np.float32)
            b = A_raw @ x
            all_samples.append((
                torch.tensor(b, dtype=torch.float32),
                torch.tensor(x, dtype=torch.float32),
                matrix_idx 
            ))

    split_idx = int(num_matrices * train_ratio)
    train_samples = []
    val_samples = []
    for sample in all_samples:
        b_tensor, x_tensor, global_idx = sample
        if global_idx < split_idx:
            train_samples.append( (b_tensor, x_tensor, global_idx) )
        else:
            val_local_idx = global_idx - split_idx
            val_samples.append( (b_tensor, x_tensor, val_local_idx) )

    return (
        all_matrices_raw[:split_idx],   # train_raw
        all_matrices_raw[split_idx:],    # val_raw
        all_matrices_norm[:split_idx],   # train_norm
        all_matrices_norm[split_idx:],   # val_norm
        train_samples,
        val_samples
    )

def generate_test_data(n=100, h=1e-3):
    A = generate_clean_fourth_order_matrix(n, h)
    x = np.random.randn(n)
    b = A @ x
    return A, b

def load_stand_data(stand_dir, max_matrices=50, num_samples_per_matrix=20):
    all_samples = []
    matrix_paths = glob.glob(os.path.join(stand_dir, "*.npz"))[:max_matrices]
    
    for path in matrix_paths:
        with np.load(path) as data:
            K = sp.coo_matrix(
                (data["A_values"], (data["A_indices"][0], data["A_indices"][1]))
            ).toarray().astype(np.float32)
            x = data["x"].astype(np.float32)
            b = data["b"].astype(np.float32)
            
            # 校验矩阵与向量的维度一致性
            assert K.shape[0] == K.shape[1], "Matrix must be square"
            assert K.shape[0] == b.size, "Matrix dimension must match b size"
            assert K.shape[0] == x.size, "Matrix dimension must match x size"
            
            K_norm = normalize_matrix(K)
                
            # 存储样本
            for _ in range(num_samples_per_matrix):
                sample = (
                    torch.tensor(b, dtype=torch.float32),
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(K, dtype=torch.float32),
                    torch.tensor(K_norm, dtype=torch.float32)
                )
                all_samples.append(sample)
    
    return all_samples

def load_stand_train_data(stand_train_dir, max_matrices=50, num_samples_per_matrix=20):
    return load_stand_data(stand_train_dir, max_matrices, num_samples_per_matrix)

def load_stand_test_data(stand_test_dir, max_matrices=10, num_samples_per_matrix=20):
    return load_stand_data(stand_test_dir, max_matrices, num_samples_per_matrix)