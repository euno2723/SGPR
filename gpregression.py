from scipy.linalg import solve_triangular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linalg_util import customized_cholesky
from gp_util import RBF, Posterior, plot_gp




class GPR:
    def __init__(self, X_train, y_train, kernel=None, noise=1.):
        self.X_train = X_train
        self.y_train = y_train
        self.noise = noise
        
        if kernel is None:
            kernel = RBF()
            self.kernel = kernel
            
        self._K = self.kernel(self.X_train)
        
        tmp = X_train.ravel()
        ext = np.abs(tmp.min() - tmp.max()) * 0.1
        self._X = np.linspace(tmp.min() - ext, 
                              tmp.max() + ext, 
                              200).reshape(-1,1)
        
        self.posterior = self._inference_posterior

    
    def _inference_posterior(self):
        m = 0 # If other mean functions are required, we need some modification here.
        
        Ky = self._K.copy()
        Ky += (np.eye(Ky.shape[0]) * (self.noise+1e-8)) # unclear why 1e-8 is added. just followed the way of GPy
        LW = customized_cholesky(Ky)
        alpha = solve_triangular(LW.T, solve_triangular(LW, self.y-m, lower=True))
        
        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=self._K)
        

    def predict(self, X_new):
        return self.posterior._raw_predict(self.kernel, X_new, self.X_train)


    def plot_sampled_prior(self, size=None):
        plt.figure()
        mu = np.zeros(self._X.shape[0])
        cov = self.kernel.K(self._X)
        samples = np.random.multivariate_normal(mu, cov, size=size)
        plot_gp(mu, cov, self._X, samples=samples)
        plt.show()

        
    def plot_sampled_posterior(self, size=None):
        plt.figure()
        mu, cov, samples = self.posterior.sampling(self.kernel, self._X, size=size)
        plot_gp(mu, cov, self.X_train, self.Y_train, samples=samples)
        plt.show()




# def main():
#     import numpy as np
#     # import pandas as pd
#     import pods

#     np.random.seed(seed=1)

#     data = pods.datasets.olympic_100m_men()
#     X, Y = data["X"], data["Y"]
#     # X_pred = np.linspace(X[:, 0].min() - 30,
#     #                      X[:, 0].max() + 30,
#     #                      500).reshape(-1, 1)

#     k = RBF(variance=1., lengthscale=10.)
#     m = GPR(X, Y)
#     m.plot_sampled_prior(size=10)


# if __name__ == '__main__':
#     main()



        
        