import numpy as np
# import pandas as pd
import GPy
import pods


data = pods.datasets.olympic_100m_men()
X, Y = data["X"], data["Y"]
X_pred = np.linspace(X[:,0].min() - 30,
                     X[:,0].max() + 30,
                     500).reshape(-1,1)

kernel = GPy.kern.RBF(input_dim=1,
                      variance=1.,
                      lengthscale=10)

model = GPy.models.GPRegression(X, Y, kernel)
# model.optimize("bfgs", max_iters=200)
mean, var = model.predict(X_pred, full_cov=True)



def main():
    print(type(kernel.K))
    print()
    print("Means:")
    print(mean)
    print()
    print("Variances:")
    print(var)


if __name__ == "__main__":
    main()