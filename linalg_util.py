import numpy as np
# from scipy.linalg import cholesky, LinAlgError
from scipy import linalg


def customized_cholesky(matrix, max_tries=5):
    matrix = np.ascontiguousarray(matrix)
    diag_matrix = np.diag(matrix)
    jitter = diag_matrix.mean() * 1e-6
    num_tries = 0
    
    try:
        L = linalg.cholesky(matrix, lower=True)
        return L
    except linalg.LinAlgError:
        num_tries += 1
        
    while num_tries <= max_tries and np.isfinite(jitter):
        try:
            L = linalg.cholesky(matrix + np.eye(matrix.shape[0]) * jitter,
                        lower=True)
            return L
        except linalg.LinAlgError:
            jitter *= 10
            num_tries += 1
            
    raise linalg.LinAlgError("Matrix is not positive definite, even with jitter.")


def symmetrify_matrix(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]
    return A



# 実験用データ作成
def generate_pd_matrix(size=None):
    if size is None:
        size = 20
    A = np.random.randn(size, 100)
    return A @ A.T

    
def generate_non_pd_matrix(size=None):    
    A = generate_pd_matrix(size=size)
    # Compute Eigdecomp
    vals, vectors = np.linalg.eig(A)
    # Set smallest eigenval to be negative with 5 rounds worth of jitter
    vals[vals.argmin()] = 0
    default_jitter = 1e-6 * np.mean(vals)
    vals[vals.argmin()] = -default_jitter * (10 ** 3.5)
    A_corrupt = (vectors * vals).dot(vectors.T)
    return A_corrupt


