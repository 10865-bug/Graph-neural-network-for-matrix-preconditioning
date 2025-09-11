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

def _tri_element_matrices(p1, p2, p3, pv_coeff=0.0):
    """
    对单个三角形，返回局部刚度K_e、质量M_e，并合成 A_e = K_e + pv_coeff * M_e
    p1,p2,p3: 3个顶点坐标 (x,y) 的 ndarray
    说明：
      - 本函数保持原样，仍返回二阶意义下的 K_e、M_e 与 A_e；
      - 在全局装配阶段，不再直接累加 A_e，而是先分别装配 K 与 M，
        再构造四阶离散 A = K M_lump^{-1} K + pv_coeff * M。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area <= 0:
        # 退化，返回零
        Z = np.zeros((3, 3), dtype=np.float32)
        return Z, Z, Z

    # P1 形函数梯度（常数）
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
    grad = np.vstack([b, c]) / (2.0 * area)  # 2 x 3

    # 刚度: K_e(i,j) = area * grad(phi_i)·grad(phi_j)
    K_e = area * (grad.T @ grad)  # 3x3

    # 质量: M_e = (area/12) * [[2,1,1],[1,2,1],[1,1,2]]
    M_e = (area / 12.0) * (2.0 * np.eye(3) + np.ones((3, 3)) - np.eye(3))

    A_e = K_e + pv_coeff * M_e
    return K_e.astype(np.float32), M_e.astype(np.float32), A_e.astype(np.float32)

def _assemble_p1(nodes, tris, boundary_mask, pv_coeff=0.0):
    """
    从节点、三角形连接关系装配A=K+pv*M，并施加Dirichlet边界 u=0：
    - 删除边界自由度对应的行列，得到内部自由度的 A_int
    返回:
      A_int (ndarray, float32), int_dofs (内部自由度索引, ndarray)

    修改点（保持接口不变）：
      - 不再直接装配二阶的 A=K+pv*M；
      - 先分别装配 K 与 M；
      - 对内部自由度形成 K_int、M_int；
      - 用行和集块的 M_lump^{-1} 近似 M^{-1}；
      - 构造四阶离散：A_int = K_int @ M_lump^{-1} @ K_int + pv_coeff * M_int。
    """
    n_nodes = nodes.shape[0]

    # 分别装配 K 与 M
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

    # 内部自由度（非边界）
    int_dofs = np.where(~boundary_mask)[0]
    K_int = K[int_dofs][:, int_dofs].toarray().astype(np.float32)
    M_int = M[int_dofs][:, int_dofs].toarray().astype(np.float32)

    # 质量集块（行和到对角）并取逆
    diagMl = M_int.sum(axis=1)  # (n_int,)
    eps = 1e-12
    D_inv = (1.0 / (diagMl + eps)).astype(np.float32)  # 向量形式，代表对角逆

    # 四阶离散：A_int = K_int @ diag(D_inv) @ K_int + pv_coeff * M_int
    # 逐列缩放等价于右乘 diag(D_inv)
    K_Dinv = (K_int * D_inv[None, :]).astype(np.float32)  # 每列乘以相应的 D_inv
    A_int = (K_Dinv @ K_int).astype(np.float32)
    if pv_coeff != 0.0:
        A_int = (A_int + pv_coeff * M_int).astype(np.float32)

    return A_int, int_dofs

def _normalize_dense(K):
    # 复用你已有 normalize_matrix 签名：如果已导入本模块的 normalize_matrix，就直接用，否则做个安全兜底
    try:
        from .data_generator import normalize_matrix as _nm  # 若本文件作为包使用
        return _nm(K)
    except Exception:
        # 简单谱范数归一（兜底）；建议仍优先使用你已有的 normalize_matrix
        s = np.linalg.norm(K, ord=2)
        s = s if s > 0 else 1.0
        return (K / s).astype(np.float32)

def _make_triangle_domain(nx):
    """
    右角单位三角形 Ω_T = {(x,y): x>=0, y>=0, x+y<=1}
    构造近似均匀三角网：
      - 节点: (i/nx, j/nx), i>=0, j>=0, i+j<=nx
      - 单元: 将每个小网格的下三角 (i,j)-(i+1,j)-(i,j+1) 作为单元
    返回:
      nodes: (N,2)
      tris:  (M,3) int 索引
      boundary_mask: (N,) bool, 边界节点 True
    """
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
            # 小右角三角 (i,j)-(i+1,j)-(i,j+1)
            i0 = ij_to_idx[(i, j)]
            i1 = ij_to_idx[(i + 1, j)]
            i2 = ij_to_idx[(i, j + 1)]
            tris.append([i0, i1, i2])
    tris = np.array(tris, dtype=np.int32)

    # 边界：i=0 或 j=0 或 i+j=nx
    boundary = np.zeros(len(nodes), dtype=bool)
    for (i, j), idx in ij_to_idx.items():
        if i == 0 or j == 0 or (i + j) == nx:
            boundary[idx] = True

    return nodes, tris, boundary

def generate_fem_triangle_matrix(nx=32, pv_coeff=0.0):
    """
    生成 P1 FEM 的 A（四阶离散矩阵）定义在单位右角三角形，Dirichlet 边界 u=0
    本实现：A = K M_lump^{-1} K + pv_coeff * M
    返回:
      A_int (float32, n_int x n_int)
    """
    nodes, tris, boundary_mask = _make_triangle_domain(nx)
    A_int, _ = _assemble_p1(nodes, tris, boundary_mask, pv_coeff=pv_coeff)
    return A_int

def _make_circle_domain(nr=12, ntheta0=12):
    """
    单位圆 Ω_C = {(x,y): x^2+y^2 <= 1}
    生成极坐标环带网格：
      - 半径 r_k = k/nr, k=0..nr
      - 每一环的角向节点数 ntheta_k = max(3, round(ntheta0 * r_k))
        并与相邻两环连成三角形
    返回:
      nodes: (N,2)
      tris:  (M,3)
      boundary_mask: (N,) bool, 外圈 r=1 上 True
    """
    nodes = [(0.0, 0.0)]
    ring_indices = [[0]]  # 中心点单独一"环"

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

    # 从中心到第一环的扇形三角
    first_ring = ring_indices[1]
    m1 = len(first_ring)
    for t in range(m1):
        i_center = 0
        i1 = first_ring[t]
        i2 = first_ring[(t + 1) % m1]
        tris.append([i_center, i1, i2])

    # 相邻环之间的四边形拆分成两个三角
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

    # 外边界：最后一环的所有节点
    boundary = np.zeros(len(nodes), dtype=bool)
    for idx in ring_indices[-1]:
        boundary[idx] = True

    return nodes, tris, boundary

def generate_fem_circle_matrix(nr=20, ntheta0=28, pv_coeff=0.0):
    """
    生成 P1 FEM 的 A（四阶离散矩阵）定义在单位圆，Dirichlet 边界 u=0
    本实现：A = K M_lump^{-1} K + pv_coeff * M
    返回:
      A_int (float32, n_int x n_int)
    """
    nodes, tris, boundary_mask = _make_circle_domain(nr=nr, ntheta0=ntheta0)
    A_int, _ = _assemble_p1(nodes, tris, boundary_mask, pv_coeff=pv_coeff)
    return A_int

def make_samples_from_dense_mats(mats, num_samples_per_matrix=20):
    """
    从给定的一组稠密矩阵 mats 生成训练样本 (b,x,K,K_norm) 的列表
    """
    samples = []
    for A in mats:
        n = A.shape[0]
        K = A.astype(np.float32)
        K_norm = _normalize_dense(K)
        for _ in range(num_samples_per_matrix):
            x = np.random.randn(n).astype(np.float32)  # 修复了这里的语法错误
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
    """
    生成三角形与圆形两类 FEM 系数矩阵样本，并合并为 (b,x,K,K_norm) 列表
    （此处的 FEM 矩阵已为四阶离散）
    """
    mats = []
    n_tri = int(round(num_matrices * tri_ratio))
    n_cir = num_matrices - n_tri

    for _ in range(n_tri):
        mats.append(generate_fem_triangle_matrix(nx=tri_nx, pv_coeff=pv_coeff))
    for _ in range(n_cir):
        mats.append(generate_fem_circle_matrix(nr=circ_nr, ntheta0=circ_ntheta0, pv_coeff=pv_coeff))

    return make_samples_from_dense_mats(mats, num_samples_per_matrix=num_samples_per_matrix)

def generate_clean_fourth_order_pv_matrix(n, h, pv_coeff=0.0):
    # 保持你的接口与实现假设不变：依赖你已有的 generate_clean_fourth_order_matrix
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
