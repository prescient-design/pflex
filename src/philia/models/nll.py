"""Gaussian mixture negative log likelihoods that can be evaluated, for use as
loss functions, but also generate samples when parameters are given

"""
from abc import ABC, abstractmethod
import random
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
__all__ = ['DiagonalGaussianNLL', 'FullRankGaussianNLL', 'DoubleGaussianNLL']

log_2_pi = 1.8378770664093453
log_2 = 0.6931471805599453


class BaseGaussianNLL(ABC):
    """Abstract base class to represent the Gaussian negative log likelihood
    (NLL).

    Gaussian NLLs or mixtures thereof with various forms of the covariance
    matrix inherit from this class.

    """
    def __init__(self, Y_dim, device):
        """
        Parameters
        ----------
        Y_dim : int
            number of parameters to predict
        device : torch.device object

        """
        self.Y_dim = Y_dim
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()

    def set_device(self, device):
        self.device = device
        if hasattr(self, 'tril_idx'):
            self.tril_idx = self.tril_idx.to(device)

    def seed_samples(self, sample_seed):
        """Seed the sampling for reproducibility
        Parameters
        ----------
        sample_seed : int
        """
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed(sample_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def unwhiten_back(self, sample):
        """Scale and shift back to the unwhitened state
        Parameters
        ----------
        pred : torch.Tensor
            network prediction of shape `[batch_size, n_samples, self.Y_dim]`
        Returns
        -------
        torch.Tensor
            the unwhitened pred

        """
        sample = sample*self.Y_std.unsqueeze(1) + self.Y_mean.unsqueeze(1)
        return sample

    @abstractmethod
    def slice(self, pred):
        """Slice the raw network prediction into meaningful Gaussian parameters

        Parameters
        ----------
        pred : torch.Tensor of shape `[batch_size, self.Y_dim]`
            the network prediction

        """
        return NotImplemented

    @abstractmethod
    def __call__(self, pred, target, reduce=False):
        """Evaluate the NLL. Must be overridden by subclasses.

        Parameters
        ----------
        pred : torch.Tensor
            raw network output for the predictions
        target : torch.Tensor
            Y labels

        """
        return NotImplemented

    def nll_diagonal(self, target, mu, logvar):
        """Evaluate the NLL for single Gaussian with diagonal covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        logvar : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the log of the diagonal elements of the
            covariance matrix

        Returns
        -------
        torch.Tensor of shape
            NLL values

        """
        precision = torch.exp(-logvar)
        # Loss kernel
        loss = precision * (target - mu)**2.0 + logvar
        # Restore prefactors
        loss += np.log(2.0*np.pi)
        loss *= 0.5
        return torch.sum(loss, dim=1)

    def get_prec_params(self, tril_elements):
        batch_size = tril_elements.shape[0]
        tril = torch.zeros([batch_size, self.Y_dim, self.Y_dim],
                           device=self.device, dtype=None)
        tril[:, self.tril_idx[0], self.tril_idx[1]] = tril_elements
        log_diag_tril = torch.diagonal(tril, offset=0, dim1=1, dim2=2)  # [batch_size, Y_dim]
        logdet_term = -torch.sum(log_diag_tril, dim=1)  # [batch_size,]
        tril[:, torch.eye(self.Y_dim, dtype=bool)] = torch.exp(log_diag_tril)
        prec_mat = torch.bmm(tril, torch.transpose(tril, 1, 2))  # [batch_size, Y_dim, Y_dim]
        return {'prec_mat': prec_mat, 'log_det': logdet_term}

    def nll_full_rank(self, target, mu, tril_elements, reduce=True):
        """Evaluate the NLL for a single Gaussian with a full-rank covariance
        matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
        tril_elements : torch.Tensor of shape [batch_size, Y_dim*(Y_dim + 1)//2]
        reduce : bool
            whether to take the mean across the batch

        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values

        """
        params = self.get_prec_params(tril_elements)
        prec_mat = params['prec_mat']
        logdet_term = params['log_det']
        y_diff = mu - target  # [batch_size, Y_dim]
        mahalanobis_term = 0.5*torch.sum(
            y_diff*torch.sum(prec_mat*y_diff.unsqueeze(-1), dim=-2), dim=-1)  # [batch_size,]
        loss = logdet_term + mahalanobis_term + 0.5*self.Y_dim*log_2_pi
        if reduce:
            return torch.mean(loss, dim=0)  # float
        else:
            return loss  # [batch_size,]

    def nll_mixture(self, target, mu, tril_elements,
                    mu2, tril_elements2, alpha):
        """Evaluate the NLL for a single Gaussian with a full but low-rank plus
        diagonal covariance matrix

        Parameters
        ----------
        target : torch.Tensor of shape [batch_size, Y_dim]
            Y labels
        mu : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
            for the first Gaussian
        tril_elements : torch.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix
        mu2 : torch.Tensor of shape [batch_size, Y_dim]
            network prediction of the mu (mean parameter) of the BNN posterior
            for the second Gaussian
        tril_elements2 : torch.Tensor of shape [batch_size, self.tril_len]
            network prediction of the elements in the precision matrix for the
            second Gaussian
        alpha : torch.Tensor of shape [batch_size, 1]
            network prediction of the logit of twice the weight on the second
            Gaussian

        Note
        ----
        The weight on the second Gaussian is required to be less than 0.5, to
        make the two Gaussians well-defined.

        Returns
        -------
        torch.Tensor of shape [batch_size,]
            NLL values

        """
        batch_size, _ = target.shape
        alpha = alpha.reshape(-1)
        log_w1p1 = torch.log1p(2.0*torch.exp(-alpha)) - log_2 - torch.log1p(torch.exp(-alpha)) - self.nll_full_rank(target, mu, tril_elements, reduce=False)  # [batch_size]
        log_w2p2 = -log_2 + self.logsigmoid(alpha) - self.nll_full_rank(target, mu2, tril_elements2, reduce=False)  # [batch_size]
        stacked = torch.stack([log_w1p1, log_w2p2], dim=1)
        log_nll = -torch.logsumexp(stacked, dim=1)
        return log_nll

    def sample_full_rank(self, n_samples, mu, tril_elements, as_numpy=True):
        """Sample from a single Gaussian posterior with a full-rank covariance
        matrix

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        mu : torch.Tensor of shape `[self.batch_size, self.Y_dim]`
            network prediction of the mu (mean parameter) of the BNN posterior
        tril_elements : torch.Tensor of shape `[self.batch_size, tril_len]`
            network prediction of lower-triangular matrix in the log-Cholesky
            decomposition of the precision matrix

        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples
        """
        samples = torch.zeros([self.batch_size, n_samples, self.Y_dim],
                              device=self.device)
        for b in range(self.batch_size):
            tril = torch.zeros([self.Y_dim, self.Y_dim],
                               device=self.device,
                               dtype=None)
            tril[self.tril_idx[0], self.tril_idx[1]] = tril_elements[b, :]
            log_diag_tril = torch.diagonal(tril, offset=0, dim1=0, dim2=1)
            tril[torch.eye(self.Y_dim, dtype=bool)] = torch.exp(log_diag_tril)
            prec_mat = torch.mm(tril, tril.T)  # [Y_dim, Y_dim]
            mvn = MultivariateNormal(loc=mu[b, :],
                                     precision_matrix=prec_mat)
            sample_b = mvn.sample([n_samples, ])
            samples[b, :, :] = sample_b
        # samples = self.unwhiten_back(samples)
        if as_numpy:
            return samples.detach().cpu().numpy()
        else:
            return samples


class DiagonalGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a single Gaussian with diagonal
    covariance matrix

    `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    posterior_name = 'DiagonalGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(DiagonalGaussianNLL, self).__init__(Y_dim, device)
        self.out_dim = Y_dim*2

    def __call__(self, pred, target, reduce=False):
        loss_batch = self.nll_diagonal(target, *self.slice(pred))
        if reduce:
            return loss_batch.mean()
        else:
            return loss_batch

    def slice(self, pred):
        d = self.Y_dim  # for readability
        return torch.split(pred, [d, d], dim=1)

    def set_trained_pred(self, pred):
        d = self.Y_dim  # for readability
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.logvar = pred[:, d:]
        self.cov_diag = torch.exp(self.logvar)

    def sample(self, mean, std, n_samples, sample_seed):
        """Sample from a Gaussian posterior with diagonal covariance matrix
        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None
        Returns
        -------
        np.array of shape `[batch_size, n_samples, self.Y_dim]`
            samples
        """
        self.seed_samples(sample_seed)
        self.Y_mean = mean
        self.Y_std = std
        eps = torch.randn(self.batch_size, n_samples, self.Y_dim)
        samples = eps*torch.exp(0.5*self.logvar.unsqueeze(1)) + self.mu.unsqueeze(1)
        samples = self.unwhiten_back(samples)
        samples = samples.data.cpu().numpy()
        return samples


class FullRankGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a single Gaussian with a full-rank
    covariance matrix

    See `BaseGaussianNLL.__init__` docstring for the parameter description.

    """
    posterior_name = 'FullRankGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(FullRankGaussianNLL, self).__init__(Y_dim, device)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0,
                                           device=device)  # lower-triang idx
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim + self.Y_dim*(self.Y_dim + 1)//2

    def __call__(self, pred, target, reduce=False):
        return self.nll_full_rank(target, *self.slice(pred), reduce=reduce)

    def slice(self, pred):
        d = self.Y_dim  # for readability
        return torch.split(pred, [d, self.tril_len], dim=1)

    def marginal_nll(self, pred, target, marginal_idx, reduce=False):
        """
        pred : torch.tensor
            Shape [B, out_dim] to define the full joint
        target : torch.tensor
            Shape [B,]
        marginal_idx : int
            idx of variable to get marginal nll for

        """
        mu, tril_elements = self.slice(pred)
        params = self.get_prec_params(tril_elements)
        prec_mat = params['prec_mat']
        # TODO: inversion done twice, for marginal and conditional
        cov_mat = torch.linalg.inv(prec_mat)  # TODO: make analytic since 2x2
        normal_obj = Normal(
            loc=mu[:, marginal_idx],
            scale=cov_mat[:, marginal_idx, marginal_idx])
        nll = -normal_obj.log_prob(target[:, marginal_idx]).sum()
        if reduce:
            return torch.mean(nll, dim=0)  # float
        else:
            return nll  # [batch_size,]

    def conditional_nll(self, pred, target, cond_idx, reduce=False):
        query_idx = 0 if cond_idx == 1 else 1
        mu, tril_elements = self.slice(pred)
        params = self.get_prec_params(tril_elements)
        prec_mat = params['prec_mat']
        cov_mat = torch.linalg.inv(prec_mat)  # TODO: make analytic since 2x2
        cond_mu = (
            mu[:, query_idx] + cov_mat[
                :, query_idx, cond_idx]/cov_mat[
                    :, cond_idx, cond_idx]*(
                        target[:, cond_idx] - mu[:, cond_idx])
        )  # [B,]
        cond_sig = (
            cov_mat[:, query_idx, query_idx] - (
                cov_mat[:, query_idx, cond_idx]/cov_mat[
                    :, cond_idx, cond_idx]*cov_mat[:, cond_idx, query_idx]
            )
        )
        normal_obj = Normal(loc=cond_mu, scale=cond_sig)
        nll = -normal_obj.log_prob(target[:, query_idx])
        if reduce:
            return torch.mean(nll, dim=0)  # float
        else:
            return nll  # [batch_size,]

    def set_trained_pred(self, pred):
        d = self.Y_dim  # for readability
        self.batch_size = pred.shape[0]
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:self.out_dim]

    def sample(self, n_samples, sample_seed, as_numpy=True):
        self.seed_samples(sample_seed)
        return self.sample_full_rank(
            n_samples, self.mu, self.tril_elements, as_numpy)


class DoubleGaussianNLL(BaseGaussianNLL):
    """The negative log likelihood (NLL) for a mixture of two Gaussians, each
    with a full but constrained as low-rank plus diagonal covariance

    Only rank 2 is currently supported. `BaseGaussianNLL.__init__` docstring
    for the parameter description.

    """
    posterior_name = 'DoubleGaussianBNNPosterior'

    def __init__(self, Y_dim, device):
        super(DoubleGaussianNLL, self).__init__(Y_dim, device)
        self.tril_idx = torch.tril_indices(self.Y_dim, self.Y_dim, offset=0,
                                           device=device)  # lower-triang idx
        self.tril_len = len(self.tril_idx[0])
        self.out_dim = self.Y_dim**2 + 3*self.Y_dim + 1

    def __call__(self, pred, target, reduce=False):
        loss_batch = self.nll_mixture(target, *self.slice(pred))
        if reduce:
            return loss_batch.mean()
        else:
            return loss_batch

    def slice(self, pred):
        d = self.Y_dim  # for readability
        return torch.split(pred, [d, self.tril_len, d, self.tril_len, 1], dim=1)

    def set_trained_pred(self, pred):
        d = self.Y_dim  # for readability
        self.batch_size = pred.shape[0]
        # First gaussian
        self.mu = pred[:, :d]
        self.tril_elements = pred[:, d:d+self.tril_len]
        self.mu2 = pred[:, d+self.tril_len:2*d+self.tril_len]
        self.tril_elements2 = pred[:, 2*d+self.tril_len:-1]
        self.w2 = 0.5*self.sigmoid(pred[:, -1].reshape(-1, 1))

    def sample(self, mean, std, n_samples, sample_seed):
        """Sample from a mixture of two Gaussians, each with a full covariance

        Parameters
        ----------
        n_samples : int
            how many samples to obtain
        sample_seed : int
            seed for the samples. Default: None
        Returns
        -------
        np.array of shape `[self.batch_size, n_samples, self.Y_dim]`
            samples

        """
        self.seed_samples(sample_seed)
        self.Y_mean = mean
        self.Y_std = std
        samples = torch.zeros([self.batch_size, n_samples, self.Y_dim],
                              device=self.device)
        # Determine first vs. second Gaussian
        unif2 = torch.rand(self.batch_size, n_samples, device=self.device)
        second_gaussian = (self.w2 > unif2)
        # Sample from second Gaussian
        samples2 = self.sample_full_rank(n_samples, self.mu2,
                                         self.tril_elements2, as_numpy=False)
        samples[second_gaussian, :] = samples2[second_gaussian, :]
        # Sample from first Gaussian
        samples1 = self.sample_full_rank(n_samples, self.mu,
                                         self.tril_elements, as_numpy=False)
        samples[~second_gaussian, :] = samples1[~second_gaussian, :]
        samples = samples.data.cpu().numpy()
        return samples
