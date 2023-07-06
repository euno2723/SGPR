from scipy.linalg import solve_triangular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linalg_util import customized_cholesky
from gp_util import RBF, Posterior




class GPR:
    def __init__(self, X, y, kernel=None, noise=1.):
        self.X = X
        self.y = y
        self.noise = noise
        
        if kernel is None:
            kernel = RBF()
            self.kernel = kernel
            
        self._K = self.kernel(self.X)


    def _inference_posterior(self):
        m = 0 # If other mean functions are required, we need some modification here.
        
        Ky = self._K.copy()
        Ky += (np.eye(Ky.shape[0]) * (self.noise+1e-8)) # unclear why 1e-8 is added. just followed the way of GPy
        
        LW = customized_cholesky(Ky)
        
        alpha = solve_triangular(LW.T, solve_triangular(LW, self.y-m, lower=True))
        
        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=self._K)
        

    def predict(self):
        pass


    def plot_sampled_prior(self, size=None):
        plt.figure()

        ext = np.abs(self.X.min() - self.X.max()) * 0.1
        X = np.linspace(self.X.min() - ext,
                        self.X.max() + ext,
                        200).reshape(-1,1)
        K = self.kernel.K(X)
        L = customized_cholesky(K)
        n = X.shape[0]
        samples = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=size) # generate x from N(0, I)

        if size == None:
            sample = L @ samples
            plt.plot(X.ravel(), sample, lw=1, ls='--')
        else:
            samples = [L @ sample for sample in samples] # y = L @ x
            for sample in samples:
                plt.plot(X.ravel(), sample, lw=1, ls='--')

        plt.show()

        
    def plot_sampled_posterior(self, size=None):
        plt.figure()
        
        plt.show()


    def plot_prediction(self)
        






























# class GPR:
#     def __init__(self, X, y, kernel=None, mean_function=None):
#         self.X = X
#         self.y = y
        
#         if kernel is None:
#             kernel = RBF()
#             self.kernel = kernel

#         if mean_function is None:
#             self.mean_function = np.zeros(self.X.shape[0]).reshape(-1,1)

#         # self.K = self.kernel.K(self.X)
#         self.mean = np.zeros(self.X.shape[0])

            
#     def plot_sampled_prior(self, size=None):
#         plt.figure()

#         extension = np.abs(self.X.min() - self.X.max()) * 0.1
#         X = np.linspace(self.X.min() - extension,
#                         self.X.max() + extension,
#                         200).reshape(-1,1)
#         K = self.kernel.K(X)
#         L = cholesky(K)
#         n = X.shape[0]
#         samples = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=size) # generate x from N(0, I)

#         if size == None:
#             sample = L @ samples
#             plt.plot(X.ravel(), sample, lw=1, ls='--')
#         else:
#             samples = [L @ sample for sample in samples] # y = L @ x
#             for sample in samples:
#                 plt.plot(X.ravel(), sample, lw=1, ls='--')

#         plt.show()



# class Posterior:
#     def __init__(self, woodbury_chol=None, woodbury_vector=None, K=None, mean=None, cov=None, K_chol=None,
#                  woodbury_inv=None, prior_mean=0):
#         """
#         woodbury_chol : a lower triangular matrix L that satisfies posterior_covariance = K - K L^{-T} L^{-1} K
#         woodbury_vector : a matrix (or vector, as Nx1 matrix) M which satisfies posterior_mean = K M
#         K : the proir covariance (required for lazy computation of various quantities)
#         mean : the posterior mean
#         cov : the posterior covariance

#         Not all of the above need to be supplied! You *must* supply:

#           K (for lazy computation)
#           or
#           K_chol (for lazy computation)

#        You may supply either:

#           woodbury_chol
#           woodbury_vector

#         Or:

#           mean
#           cov

#         Of course, you can supply more than that, but this class will lazily
#         compute all other quantites on demand.

#         """
#         # obligatory
#         self._K = K

#         if ((woodbury_chol is not None) and (woodbury_vector is not None)) \
#                 or ((woodbury_inv is not None) and (woodbury_vector is not None)) \
#                 or ((woodbury_inv is not None) and (mean is not None)) \
#                 or ((mean is not None) and (cov is not None)):
#             pass  # we have sufficient to compute the posterior
#         else:
#             raise ValueError("insufficient information to compute the posterior")

#         self._K_chol = K_chol
#         self._K = K
#         # option 1:
#         self._woodbury_chol = woodbury_chol
#         self._woodbury_vector = woodbury_vector

#         # option 2.
#         self._woodbury_inv = woodbury_inv
#         # and woodbury vector

#         # option 2:
#         self._mean = mean
#         self._covariance = cov
#         self._prior_mean = prior_mean

#         # compute this lazily
#         self._precision = None

#     @property
#     def mean(self):
#         """
#         Posterior mean
#         $$
#         K_{xx}v
#         v := \texttt{Woodbury vector}
#         $$
#         """
#         if self._mean is None:
#             self._mean = np.dot(self._K, self.woodbury_vector)
#         return self._mean

#     @property
#     def covariance(self):
#         """
#         Posterior covariance
#         $$
#         K_{xx} - K_{xx}W_{xx}^{-1}K_{xx}
#         W_{xx} := \texttt{Woodbury inv}
#         $$
#         """
#         if self._covariance is None:
#             # LiK, _ = dtrtrs(self.woodbury_chol, self._K, lower=1)
#             self._covariance = (
#             np.atleast_3d(self._K) - np.tensordot(np.dot(np.atleast_3d(self.woodbury_inv).T, self._K), self._K,
#                                                   [1, 0]).T).squeeze()
#             # self._covariance = self._K - self._K.dot(self.woodbury_inv).dot(self._K)
#         return self._covariance

#     @property
#     def woodbury_chol(self):
#         """
#         return $L_{W}$ where L is the lower triangular Cholesky decomposition of the Woodbury matrix
#         $$
#         L_{W}L_{W}^{\top} = W^{-1}
#         W^{-1} := \texttt{Woodbury inv}
#         $$
#         """
#         if self._woodbury_chol is None:
#             # compute woodbury chol from
#             if self._woodbury_inv is not None:
#                 winv = np.atleast_3d(self._woodbury_inv)
#                 self._woodbury_chol = np.zeros(winv.shape)
#                 for p in range(winv.shape[-1]):
#                     self._woodbury_chol[:, :, p] = pdinv(winv[:, :, p])[2]
#                     # Li = jitchol(self._woodbury_inv)
#                     # self._woodbury_chol, _ = dtrtri(Li)
#                     # W, _, _, _, = pdinv(self._woodbury_inv)
#                     # symmetrify(W)
#                     # self._woodbury_chol = jitchol(W)
#             # try computing woodbury chol from cov
#             elif self._covariance is not None:
#                 raise NotImplementedError("TODO: check code here")
#                 B = self._K - self._covariance
#                 tmp, _ = dpotrs(self.K_chol, B)
#                 self._woodbury_inv, _ = dpotrs(self.K_chol, tmp.T)
#                 _, _, self._woodbury_chol, _ = pdinv(self._woodbury_inv)
#             else:
#                 raise ValueError("insufficient information to compute posterior")
#         return self._woodbury_chol

#     @property
#     def woodbury_inv(self):
#         """
#         The inverse of the woodbury matrix, in the gaussian likelihood case it is defined as
#         $$
#         (K_{xx} + \Sigma_{xx})^{-1}
#         \Sigma_{xx} := \texttt{Likelihood.variance / Approximate likelihood covariance}
#         $$
#         """
#         if self._woodbury_inv is None:
#             if self._woodbury_chol is not None:
#                 self._woodbury_inv, _ = dpotri(self._woodbury_chol, lower=1)
#                 # self._woodbury_inv, _ = dpotrs(self.woodbury_chol, np.eye(self.woodbury_chol.shape[0]), lower=1)
#                 symmetrify(self._woodbury_inv)
#             elif self._covariance is not None:
#                 B = np.atleast_3d(self._K) - np.atleast_3d(self._covariance)
#                 self._woodbury_inv = np.empty_like(B)
#                 for i in range(B.shape[-1]):
#                     tmp, _ = dpotrs(self.K_chol, B[:, :, i])
#                     self._woodbury_inv[:, :, i], _ = dpotrs(self.K_chol, tmp.T)
#         return self._woodbury_inv

#     @property
#     def woodbury_vector(self):
#         """
#         Woodbury vector in the gaussian likelihood case only is defined as
#         $$
#         (K_{xx} + \Sigma)^{-1}Y
#         \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
#         $$
#         """
#         if self._woodbury_vector is None:
#             self._woodbury_vector, _ = dpotrs(self.K_chol, self.mean - self._prior_mean)
#         return self._woodbury_vector

#     @property
#     def K_chol(self):
#         """
#         Cholesky of the prior covariance K
#         """
#         if self._K_chol is None:
#             self._K_chol = jitchol(self._K)
#         return self._K_chol

#     def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):
#         woodbury_vector = self.woodbury_vector
#         woodbury_inv = self.woodbury_inv

#         if not isinstance(Xnew, VariationalPosterior):
#             Kx = kern.K(pred_var, Xnew)
#             mu = np.dot(Kx.T, woodbury_vector)
#             if len(mu.shape) == 1:
#                 mu = mu.reshape(-1, 1)
#             if full_cov:
#                 Kxx = kern.K(Xnew)
#                 if woodbury_inv.ndim == 2:
#                     var = Kxx - np.dot(Kx.T, np.dot(woodbury_inv, Kx))
#                 elif woodbury_inv.ndim == 3:  # Missing data
#                     var = np.empty((Kxx.shape[0], Kxx.shape[1], woodbury_inv.shape[2]))
#                     from ...util.linalg import mdot
#                     for i in range(var.shape[2]):
#                         var[:, :, i] = (Kxx - mdot(Kx.T, woodbury_inv[:, :, i], Kx))
#                 var = var
#             else:
#                 Kxx = kern.Kdiag(Xnew)
#                 if woodbury_inv.ndim == 2:
#                     var = (Kxx - np.sum(np.dot(woodbury_inv.T, Kx) * Kx, 0))[:, None]
#                 elif woodbury_inv.ndim == 3:  # Missing data
#                     var = np.empty((Kxx.shape[0], woodbury_inv.shape[2]))
#                     for i in range(var.shape[1]):
#                         var[:, i] = (Kxx - (np.sum(np.dot(woodbury_inv[:, :, i].T, Kx) * Kx, 0)))
#                 var = var
#                 var = np.clip(var, 1e-15, np.inf)
#         else:
#             psi0_star = kern.psi0(pred_var, Xnew)
#             psi1_star = kern.psi1(pred_var, Xnew)
#             psi2_star = kern.psi2n(pred_var, Xnew)
#             la = woodbury_vector
#             mu = np.dot(psi1_star, la)  # TODO: dimensions?
#             N, M, D = psi0_star.shape[0], psi1_star.shape[1], la.shape[1]

#             if full_cov:
#                 raise NotImplementedError(
#                     "Full covariance for Sparse GP predicted with uncertain inputs not implemented yet.")
#                 var = np.zeros((Xnew.shape[0], la.shape[1], la.shape[1]))
#                 di = np.diag_indices(la.shape[1])
#             else:
#                 tmp = psi2_star - psi1_star[:, :, None] * psi1_star[:, None, :]
#                 var = (tmp.reshape(-1, M).dot(la).reshape(N, M, D) * la[None, :, :]).sum(1) + psi0_star[:, None]
#                 if woodbury_inv.ndim == 2:
#                     var += -psi2_star.reshape(N, -1).dot(woodbury_inv.flat)[:, None]
#                 else:
#                     var += -psi2_star.reshape(N, -1).dot(woodbury_inv.reshape(-1, D))
#                 var = np.clip(var, 1e-15, np.inf)
#         return mu, var







def main():
    import numpy as np
    # import pandas as pd
    import pods

    np.random.seed(seed=1)

    data = pods.datasets.olympic_100m_men()
    X, Y = data["X"], data["Y"]
    # X_pred = np.linspace(X[:, 0].min() - 30,
    #                      X[:, 0].max() + 30,
    #                      500).reshape(-1, 1)

    k = RBF(variance=1., lengthscale=10.)
    m = GPR(X, Y)
    m.plot_sampled_prior(size=10)


if __name__ == '__main__':
    main()



        
        