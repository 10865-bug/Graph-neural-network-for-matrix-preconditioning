import numpy as np
from numpy.linalg import norm

def run_gmres(A, b, M=None, tol=1e-6, m=50, maxiter=1000):
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    x0 = np.zeros_like(b, dtype=np.float64)
    r0 = b - A @ x0
    beta = norm(r0)
    if beta < tol:
        return x0, 0
    
    total_iters = 0
    for _ in range(maxiter // m):
        V = [r0 / (beta + 1e-12)]  # 防止除零
        Z = []
        H = np.zeros((m+1, m), dtype=np.float64)
        
        for j in range(m):
            if M is not None:
                zj = M(V[j].reshape(-1,1)).flatten().astype(np.float64)
            else:
                zj = V[j].copy().astype(np.float64)
            Z.append(zj)
            
            w = A @ zj
            
            # 修正正交化过程
            for i in range(j+1):
                H[i, j] = np.dot(w, V[i])
                w -= H[i, j] * V[i]
            
            h_norm = norm(w)
            if h_norm < 1e-12:  # 提前终止避免除零
                break
            H[j+1, j] = h_norm
            V.append(w / (h_norm + 1e-12))  # 防止除零
            
        # 检查 H 的有效维度
        actual_m = j + 1  # 实际生成的Krylov子空间维度
        if actual_m == 0:
            break
            
        e1 = np.zeros(actual_m + 1, dtype=np.float64)
        e1[0] = beta
        H_sub = H[:actual_m+1, :actual_m]
        
        try:
            y, _, _, _ = np.linalg.lstsq(H_sub, e1, rcond=None)
        except np.linalg.LinAlgError:
            print("警告: 最小二乘求解失败，提前终止迭代")
            break
        
        Zm = np.column_stack(Z[:actual_m]).astype(np.float64)
        xm = x0 + Zm @ y.astype(np.float64)
        
        residual = norm(b - A @ xm)
        if residual < tol:
            return xm.astype(np.float32), total_iters + actual_m
        
        x0 = xm.copy()
        r0 = b - A @ x0
        beta = norm(r0)
        total_iters += actual_m
    
    return xm.astype(np.float32), total_iters