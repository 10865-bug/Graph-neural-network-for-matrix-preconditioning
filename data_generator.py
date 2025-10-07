import numpy as np
import scipy.sparse as sp
import torch
import glob
import os

def generate_strict_diagonally_dominant(n):
    A = np.zeros((n, n))
    
    main_diag = np.random.uniform(40, 80, n)
    A[np.diag_indices(n)] = main_diag
    
    for offset in [1, 2]:
        diag_len = n - abs(offset)
        off_diag = np.random.uniform(-10, 10, diag_len)
        np.fill_diagonal(A[offset:], off_diag)
        np.fill_diagonal(A[-offset:], off_diag)
    
    return A

def generate_clean_fourth_order_matrix(n, h):
    main = 6 / (h**4) * np.ones(n)
    off1 = -4 / (h**4) * np.ones(n-1)
    off2 = 1 / (h**4) * np.ones(n-2)
    
    diagonals = [off2, off1, main, off1, off2]
    offsets = [-2, -1, 0, 1, 2]
    A = sp.diags(diagonals, offsets, shape=(n, n)).toarray()
    return A.astype(np.float32)

def generate_test_matrices(n, num=100, h_range=[1e-2, 1e-1], only_fourth_order=False):
    test_mats = []
    
    if only_fourth_order:
        for _ in range(num):
            h = np.random.uniform(h_range[0], h_range[1])
            test_mats.append(generate_clean_fourth_order_matrix(n, h))
    else:
        num_dominant = num // 2
        for _ in range(num_dominant):
            test_mats.append(generate_strict_diagonally_dominant(n))
        
        num_diff = num - num_dominant
        for _ in range(num_diff):
            h = np.random.uniform(h_range[0], h_range[1])
            test_mats.append(generate_clean_fourth_order_matrix(n, h))
    
    return test_mats

def generate_random_pentadiagonal_matrix(n, h):
    A = np.zeros((n, n))
    
    main_diag = np.random.uniform(40, 100, n)
    np.fill_diagonal(A, main_diag)
    
    for offset in [1, 2]:
        diag_len = n - offset
        off_diag = np.random.uniform(-10, 10, diag_len)
        np.fill_diagonal(A[offset:], off_diag)    
        np.fill_diagonal(A[:, offset:], off_diag)
    
    return A

def normalize_matrix(A):
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy() 
    
    row_norm = np.max(np.sum(np.abs(A), axis=1))
    col_norm = np.max(np.sum(np.abs(A), axis=0))
    gamma = min(row_norm, col_norm) + 1e-12
    
    return A / gamma

def generate_data(num_matrices, n, num_samples_per_matrix=100, train_ratio=0.8, h_range=[1e-2, 5e-2]):
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

def generate_test_data(n=500, h=1e-0):
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
            
            assert K.shape[0] == K.shape[1], "Matrix must be square"
            assert K.shape[0] == b.size, "Matrix dimension must match b size"
            assert K.shape[0] == x.size, "Matrix dimension must match x size"
            
            K_norm = normalize_matrix(K)
                
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

def _tri_element_matrices(p1, p2, p3, pv_coeff=0.0):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area <= 0:
        Z = np.zeros((3, 3), dtype=np.float32)
        return Z, Z, Z

    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
    grad = np.vstack([b, c]) / (2.0 * area)  # 2 x 3

    K_e = area * (grad.T @ grad)  # 3x3

    M_e = (area / 12.0) * (2.0 * np.eye(3) + np.ones((3, 3)) - np.eye(3))

    A_e = K_e + pv_coeff * M_e
    return K_e.astype(np.float32), M_e.astype(np.float32), A_e.astype(np.float32)

def _assemble_p1(nodes, tris, boundary_mask, pv_coeff=0.0):
    n_nodes = nodes.shape[0]

    rows_K, cols_K, vals_K = [], [], []
    rows_M, cols_M, vals_M = [], [], []

    for tri in tris:
        i, j, k = tri
        K_e, M_e, _ = _tri_element_matrices(nodes[i], nodes[j], nodes[k], pv_coeff=0.0)
        loc = [i, j, k]
        for a in range(3):
            ia = loc[a]
            for b in range(3):
                ib = loc[b]
                rows_K.append(ia); cols_K.append(ib); vals_K.append(K_e[a, b])
                rows_M.append(ia); cols_M.append(ib); vals_M.append(M_e[a, b])

    K = sp.coo_matrix((vals_K, (rows_K, cols_K)), shape=(n_nodes, n_nodes)).tocsr()
    M = sp.coo_matrix((vals_M, (rows_M, cols_M)), shape=(n_nodes, n_nodes)).tocsr()

    int_dofs = np.where(~boundary_mask)[0]
    K_int = K[int_dofs][:, int_dofs].toarray().astype(np.float32)
    M_int = M[int_dofs][:, int_dofs].toarray().astype(np.float32)

    diagMl = M_int.sum(axis=1)  # (n_int,)
    eps = 1e-12
    D_inv = (1.0 / (diagMl + eps)).astype(np.float32)

    K_Dinv = (K_int * D_inv[None, :]).astype(np.float32)
    A_int = (K_Dinv @ K_int).astype(np.float32)
    if pv_coeff != 0.0:
        A_int = (A_int + pv_coeff * M_int).astype(np.float32)

    return A_int, int_dofs

def _normalize_dense(K):
    try:
        from .data_generator import normalize_matrix as _nm 
        return _nm(K)
    except Exception:
        s = np.linalg.norm(K, ord=2)
        s = s if s > 0 else 1.0
        return (K / s).astype(np.float32)

def _make_triangle_domain(nx):
    coords = []
    ij_to_idx = {}
    for i in range(nx + 1):
        for j in range(nx + 1 - i):
            idx = len(coords)
            coords.append((i / nx, j / nx))
            ij_to_idx[(i, j)] = idx
    nodes = np.array(coords, dtype=np.float32)

    tris = []
    for i in range(nx):
        for j in range(nx - i):
            i0 = ij_to_idx[(i, j)]
            i1 = ij_to_idx[(i + 1, j)]
            i2 = ij_to_idx[(i, j + 1)]
            tris.append([i0, i1, i2])
    tris = np.array(tris, dtype=np.int32)

    boundary = np.zeros(len(nodes), dtype=bool)
    for (i, j), idx in ij_to_idx.items():
        if i == 0 or j == 0 or (i + j) == nx:
            boundary[idx] = True

    return nodes, tris, boundary

def generate_fem_triangle_matrix(nx=32, pv_coeff=0.0):
    nodes, tris, boundary_mask = _make_triangle_domain(nx)
    A_int, _ = _assemble_p1(nodes, tris, boundary_mask, pv_coeff=pv_coeff)
    return A_int

def _make_circle_domain(nr=12, ntheta0=12):
    nodes = [(0.0, 0.0)]
    ring_indices = [[0]] 

    for k in range(1, nr + 1):
        r = k / nr
        ntheta_k = max(3, int(round(ntheta0 * r)))
        idxs = []
        for t in range(ntheta_k):
            theta = 2.0 * np.pi * t / ntheta_k
            nodes.append((r * np.cos(theta), r * np.sin(theta)))
            idxs.append(len(nodes) - 1)
        ring_indices.append(idxs)

    nodes = np.array(nodes, dtype=np.float32)
    tris = []

    first_ring = ring_indices[1]
    m1 = len(first_ring)
    for t in range(m1):
        i_center = 0
        i1 = first_ring[t]
        i2 = first_ring[(t + 1) % m1]
        tris.append([i_center, i1, i2])
        
    for k in range(1, nr):
        ring_a = ring_indices[k]
        ring_b = ring_indices[k + 1]
        ma, mb = len(ring_a), len(ring_b)
        for t in range(max(ma, mb)):
            a0 = ring_a[t % ma]
            a1 = ring_a[(t + 1) % ma]
            b0 = ring_b[t % mb]
            b1 = ring_b[(t + 1) % mb]
            tris.append([a0, b0, b1])
            tris.append([a0, b1, a1])

    tris = np.array(tris, dtype=np.int32)

    boundary = np.zeros(len(nodes), dtype=bool)
    for idx in ring_indices[-1]:
        boundary[idx] = True

    return nodes, tris, boundary

def generate_fem_circle_matrix(nr=20, ntheta0=28, pv_coeff=0.0):
    nodes, tris, boundary_mask = _make_circle_domain(nr=nr, ntheta0=ntheta0)
    A_int, _ = _assemble_p1(nodes, tris, boundary_mask, pv_coeff=pv_coeff)
    return A_int

def make_samples_from_dense_mats(mats, num_samples_per_matrix=20):
    samples = []
    for A in mats:
        n = A.shape[0]
        K = A.astype(np.float32)
        K_norm = _normalize_dense(K)
        for _ in range(num_samples_per_matrix):
            x = np.random.randn(n).astype(np.float32) 
            b = K @ x
            samples.append((
                torch.tensor(b, dtype=torch.float32),
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(K, dtype=torch.float32),
                torch.tensor(K_norm, dtype=torch.float32)
            ))
    return samples

def generate_fem_samples(num_matrices=10,
                         tri_ratio=0.5,
                         tri_nx=28,
                         circ_nr=20,
                         circ_ntheta0=28,
                         num_samples_per_matrix=20,
                         pv_coeff=0.0):
    mats = []
    n_tri = int(round(num_matrices * tri_ratio))
    n_cir = num_matrices - n_tri

    for _ in range(n_tri):
        mats.append(generate_fem_triangle_matrix(nx=tri_nx, pv_coeff=pv_coeff))
    for _ in range(n_cir):
        mats.append(generate_fem_circle_matrix(nr=circ_nr, ntheta0=circ_ntheta0, pv_coeff=pv_coeff))

    return make_samples_from_dense_mats(mats, num_samples_per_matrix=num_samples_per_matrix)

def generate_clean_fourth_order_pv_matrix(n, h, pv_coeff=0.0):
    A = generate_clean_fourth_order_matrix(n, h).astype(np.float32)
    A = A + (pv_coeff * np.eye(n, dtype=np.float32))
    return A

def generate_fourth_order_pv_matrices(n, num=100, h_range=[1e-3, 1e-1], pv_coeff=0.0):
    mats = []
    for _ in range(num):
        h = np.random.uniform(h_range[0], h_range[1])
        mats.append(generate_clean_fourth_order_pv_matrix(n, h, pv_coeff))
    return mats

def generate_fem_test_matrices(num=50, tri_ratio=0.5, tri_nx=28, circ_nr=20, circ_ntheta0=28, pv_coeff=0.0):
    mats = []
    n_tri = int(round(num * tri_ratio))
    for _ in range(n_tri):
        mats.append(generate_fem_triangle_matrix(nx=tri_nx, pv_coeff=pv_coeff))
    for _ in range(num - n_tri):
        mats.append(generate_fem_circle_matrix(nr=circ_nr, ntheta0=circ_ntheta0, pv_coeff=pv_coeff))
    return mats

def generate_test_data_pv(n=100, h=1e-3, pv_coeff=0.0):
    A = generate_clean_fourth_order_pv_matrix(n, h, pv_coeff)
    x = np.random.randn(n).astype(np.float32)
    b = A @ x
    return A, b

def generate_fem_test_data(shape='triangle', tri_nx=28, circ_nr=20, circ_ntheta0=28, pv_coeff=0.0):
    if shape == 'triangle':
        A = generate_fem_triangle_matrix(nx=tri_nx, pv_coeff=pv_coeff)
    else:
        A = generate_fem_circle_matrix(nr=circ_nr, ntheta0=circ_ntheta0, pv_coeff=pv_coeff)
    n = A.shape[0]
    x = np.random.randn(n).astype(np.float32)
    b = A @ x
    return A, b

