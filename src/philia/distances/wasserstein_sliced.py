"""
Sliced Wasserstein distance using Monte Carlo integration over random
projections

"""
import numpy as np
import torch
from scipy.special import gamma


def get_prop_const(dim, p):
    return ((2/dim)**0.5)*(gamma(0.5*dim + 0.5*p)/gamma(0.5*dim))**(1/p)


def get_wasserstein_mc_expectation(query_samples, target_samples, theta, p):
    """
    Get the sliced wasserstein metric using Monte Carlo integration given by the
    projections

    Parameters
    ----------
    query_samples, target_samples ~ [num_samples, dim]
    theta : torch.tensor
        Projections of shape [num_proj, dim]

    Returns
    -------
    torch.tensor
        ~ [num_proj,]
    """
    X_prod = torch.matmul(
        query_samples, theta.transpose(0, 1))  # [num_samples, num_proj]
    Y_prod = torch.matmul(
        target_samples, theta.transpose(0, 1))  # [num_samples, num_proj]
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
            torch.sort(X_prod, dim=0)[0]
            - torch.sort(Y_prod, dim=0)[0]
        )
    )  # [num_samples, num_proj]
    wasserstein_distance = torch.mean(
        torch.pow(wasserstein_distance, p), dim=0, keepdim=True)  # [1, num_proj]
    return wasserstein_distance


def get_gaussian_projections(dim, num_projections):
    """
    Get Gaussian random projections. Gaussian projections were shown to be
    equivalent to uniform projections on the unit hypersphere (S^{d-1}), up to a
    proportionality constant that depends on the dimension and order p.
    For p=2, the constant is 1.

    Reference
    ---------
    Nadjahi, Kimia, et al. "Fast approximation of the sliced-Wasserstein
    distance using concentration of random projections."
    NeurIPS 34 (2021): 12411-12424.

    Returns
    -------
    torch.tensor
        Projection coefficients of shape `[num_projections, dim]`
    """
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(
        torch.sum(projections**2, dim=1, keepdim=True))
    return projections


def get_gaussian_sliced_p_wasserstein_from_samples(
        query_samples: torch.tensor,
        target_samples: torch.tensor,
        num_projections : int = 10,
        p: float = 2.0):
    """
    query_samples, target_samples ~ [num_samples, dim]
    theta ~ [num_proj, dim]

    Returns
    -------
    float
        Sliced Wasserstein with integration over the unit sphere

    """
    # FIXME jwp: actually enforce typing
    if isinstance(query_samples, np.ndarray):
        query_samples = torch.tensor(query_samples)
    if isinstance(target_samples, np.ndarray):
        target_samples = torch.tensor(target_samples)
    dim = query_samples.size(1)
    theta = get_gaussian_projections(
        dim, num_projections).to(query_samples.device)  # [num_proj, dim]
    theta = theta.to(query_samples.dtype).to(query_samples.device)
    sw = get_wasserstein_mc_expectation(
        query_samples, target_samples, theta, p=p).mean()
    # [num_proj, 1] -> mean -> [1,]
    return torch.pow(sw, 1./p).item()
