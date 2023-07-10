from gpr import GPR

class SGPR(GPR):
    def __init__(self, X_train, y_train, kernel=None, noise=1.):
        super().__init__(X_train, y_train, kernel, noise)