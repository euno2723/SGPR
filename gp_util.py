import numpy as np
from scipy.linalg.lapack import dpotri
from linalg_util import *


# radial baisis kernel
class RBF:
    def __init__(self, variance=1., lengthscale=0.1):
        self.variance=variance
        self.lengthscale=lengthscale
        
    def K(self, X, X2=None):
        return self.variance * np.exp(-0.5 * (self._euc_dist(X, X2) / self.lengthscale)**2)
        
    def _euc_dist(self, X, X2):
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*(np.dot(X, X.T)) + (Xsq[:,None] + Xsq[None,:]) 
            r2 = np.clip(r2, 0, np.inf)
            np.fill_diagonal(r2, 0.)
            return np.sqrt(r2)
        else:
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)


# common posterior distribution 
class Posterior:
    def __init__(self, woodbury_chol=None, woodbury_vector=None, woodbury_inv=None, K=None, K_chol=None):
        self._K_chol = K_chol
        self._K = K
        self._woodbury_chol = woodbury_chol
        self._woodbury_vector = woodbury_vector
        self._woodbury_inv = woodbury_inv
        
        self._mean = None
        self._covariance = None

    @property
    def woodbury_inv(self):
        if self._woodbury_inv is None:
            if self._woodbury_chol is not None:
                self._woodbury_inv, _ = dpotri(self._woodbury_chol, lower=True)
                symmetrify_matrix(self._woodbury_inv)
        return self._woodbury_inv
    
    @property
    def mean(self): # posterior mean
        if self._mean is None:
            self._mean = self._K @ self.woodbury_vector
        return self._mean

    @property
    def covariance(self): # posterior covariance
        if self._covariance is None:
            self._covariance = self._K - self._K @ self.woodbury_inv @ self._K
        return self._covariance
    
    
    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = customized_cholesky(self._K)
        return self._K_chol

    def _raw_predict(self, kern, Xnew, pred_var):
        woodbury_vector = self.woodbury_vector
        woodbury_inv = self.woodbury_inv

        Kx = kern.K(pred_var, Xnew)
        mu = Kx.T @ woodbury_vector
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        Kxx = kern.K(Xnew)
        var = Kxx - Kx.T @ woodbury_inv @ Kx

        return mu, var