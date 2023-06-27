from scipy.linalg import cholesky, solve_triangular
import numpy as np
import pandas as pd

from linalg_util import custom_cholesky, generate_non_pd_matrix
from kernel import RBF


class GPR:
    def __init__(self, X, y, kernel=None):
        self.X = X
        self.y = y
        self.kernel = None
        
        if self.kernel is None:
            kernel = RBF()
            self.kernel = kernel
            
    def sample_from_prior(self):
        
        