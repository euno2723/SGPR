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