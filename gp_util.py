import numpy as np
from scipy.linalg.lapack import dpotri
from linalg_util import *
import matplotlib.pyplot as plt


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
        

    @property
    def woodbury_inv(self):
        if self._woodbury_inv is None:
            if self._woodbury_chol is not None:
                self._woodbury_inv, _ = dpotri(self._woodbury_chol, lower=True)
                symmetrify_matrix(self._woodbury_inv)
        return self._woodbury_inv
    
    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = customized_cholesky(self._K)
        return self._K_chol

    def _raw_predict(self, kernel, X_new, pred_var):
        woodbury_vector = self.woodbury_vector
        woodbury_inv = self.woodbury_inv

        Kx = kernel.K(pred_var, X_new)
        mu = Kx.T @ woodbury_vector
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        Kxx = kernel.K(X_new)
        cov = Kxx - Kx.T @ woodbury_inv @ Kx

        return mu, cov

    def sampling(self, kernel, X_new, pred_var, size=None):
        mu, cov = self._raw_predict(kernel, X_new, pred_var)
        samples = np.random.multivariate_normal(mu, cov, size=size)
        return mu, cov, samples
        
    
# 可視化
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()



# 実験用データ作成
import pods

def make_data():
    data = pods.datasets.olympic_100m_men()
    X, Y = data["X"], data["Y"]
    X_min, X_max = X[:,0].min(), X[:,0].max()
    X_min_max_diff = X_max - X_min
    X_pred = np.linspace(X[:,0].min() - X_min_max_diff / 10,
                        X[:,0].max() + X_min_max_diff / 10,
                        500).reshape(-1,1)
    return X, Y, X_pred