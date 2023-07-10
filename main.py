from gp_util import RBF, make_data
from gpr import GPR



def main():
    X, Y, X_pred = make_data()

    k = RBF(variance=1., lengthscale=10.)
    m = GPR(X, Y, kernel=k)
    # m.plot_sampled_prior(size=10)
    # m.plot_sampled_prior()
    m.plot_sampled_posterior()



if __name__ == '__main__':
    main()