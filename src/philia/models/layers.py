import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO


class CustomVariationalELBO(VariationalELBO):
    """Subclass VariationalELBO to disable sum reduction,
    so that we can mask out certain examples from each batch

    """
    def __init__(self, *args, **kwargs):
        super(CustomVariationalELBO, self).__init__(*args, **kwargs)

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)


class StochasticVariationalGP(ApproximateGP):
    def __init__(self, d_embed, n_train):
        self.n_train = n_train
        inducing_points = torch.randn(500, d_embed)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(StochasticVariationalGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.normalizer = torch.nn.BatchNorm1d(num_features=d_embed, affine=False)

    def forward(self, x):
        x = self.normalizer(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def set_to_train(self):
        self.train()
        self.likelihood.train()

    def set_to_eval(self):
        self.eval()
        self.likelihood.eval()


