import numpy as np
# from scipy.linalg import cholesky, LinAlgError
from scipy import linalg


def cholesky(matrix, max_tries=5):
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

    
    
def generate_non_pd_matrix():    
    # Create PD matrix
    A = np.random.randn(20, 100)
    A = A.dot(A.T)
    # Compute Eigdecomp
    vals, vectors = np.linalg.eig(A)
    # Set smallest eigenval to be negative with 5 rounds worth of jitter
    vals[vals.argmin()] = 0
    default_jitter = 1e-6 * np.mean(vals)
    vals[vals.argmin()] = -default_jitter * (10 ** 3.5)
    A_corrupt = (vectors * vals).dot(vectors.T)
    return A_corrupt
