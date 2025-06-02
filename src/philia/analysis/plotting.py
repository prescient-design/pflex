from __future__ import division
import io
import os
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import gaussian_kde
from philia.trainval_data.transforms import (
    inverse_scale_dist,
    inverse_scale_dist_log,
    inverse_scale_angles,
    inverse_scale_dist_std,
    inverse_scale_dist_std_log,
    inverse_scale_angles_std,
    inverse_scale_phi,
    inverse_scale_psi)


def buffer_plot_and_get(fig):
    # https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def plot_recovery_dist(true, pred_mean, pred_std=None, inverse_scale=False, to_PIL=True, scaling='scalar'):
    plt.close('all')
    fig, ax = plt.subplots()
    if inverse_scale:
        if scaling == 'scalar':
            true = inverse_scale_dist(true)
            pred_mean = inverse_scale_dist(pred_mean)
        elif 'log' in scaling:
            true = inverse_scale_dist_log(true)
            pred_mean = inverse_scale_dist_log(pred_mean)
        if pred_std is not None:
            if scaling == 'scalar':
                pred_std = inverse_scale_dist_std(pred_std)
            elif 'log' in scaling:
                pred_std = inverse_scale_dist_std_log(pred_std)
    if pred_std is None:
        n_std = np.abs((pred_mean - true)/pred_std)
        ax.scatter(true, pred_mean, color='tab:blue', marker='.', alpha=0.3)
    else:
        n_std = np.abs((pred_mean - true)/pred_std)
        points = ax.scatter(true, pred_mean, c=n_std, marker='.', alpha=0.3)
        cbar = fig.colorbar(points)
        cbar.set_label('Number of predicted STD away from truth')
    ax.plot(true, true, color='tab:gray', linestyle='--')
    ax.set_xlabel('True / Ang')
    ax.set_ylabel('Pred mean / Ang')

    if to_PIL:
        return buffer_plot_and_get(fig)
    else:
        return fig


def plot_recovery_angles(true, pred_mean, pred_std=None, inverse_scale=False, item = 'phi', to_PIL=True):
    plt.close('all')
    fig, ax = plt.subplots()
    if inverse_scale:
        if item == 'phi':
            true = inverse_scale_phi(true)
            pred_mean = inverse_scale_phi(pred_mean)
        if item == 'psi':
            true = inverse_scale_psi(true)
            pred_mean = inverse_scale_psi(pred_mean)
        # true = inverse_scale_angles(true)
        # pred_mean = inverse_scale_angles(pred_mean)
        # if pred_std is not None:
        #     pred_std = inverse_scale_angles_std(pred_std)
    # if inverse_scale_joint:
        if pred_std is not None:
            pred_std = inverse_scale_angles_std(pred_std)
    # if inverse_scale_psi:
    #     true = inverse_scale_psi(true)
    #     pred_mean = inverse_scale_psi(pred_mean)
    #     if pred_std is not None:
    #         pred_std = inverse_scale_angles_std(pred_std)
    if pred_std is None:
        n_std = np.abs((pred_mean - true)/pred_std)
        ax.scatter(true, pred_mean, color='tab:blue', marker='.', alpha=0.3)
    else:
        n_std = np.abs((pred_mean - true)/pred_std)
        points = ax.scatter(true, pred_mean, c=n_std, marker='.', alpha=0.3)
        cbar = fig.colorbar(points)
        cbar.set_label('Number of predicted STD away from truth')
    ax.plot(true, true, color='tab:gray', linestyle='--')
    ax.set_xlabel('True / deg')
    ax.set_ylabel('Pred mean / deg')
    # ax.set_xlim([-180, 180])
    # ax.set_ylim([-180, 180])
    if to_PIL:
        return buffer_plot_and_get(fig)
    else:
        return fig


# def get_stats(batch, out, item, inverse_scale=False, remove_nan=False):
#     """
#     Get inference results for the given batch

#     Parameters
#     ----------
#     inverse_scale : bool
#         Whether to inverse transform the quantities back to their physical units
#     remove_nan : bool
#         Whether to remove distance or angle elements that had nan labels

#     """
#     # Get label if available
#     true = batch[f'{item}'].reshape(-1) if item in batch else None
#     if remove_nan:
#         is_nan = batch[f'flag_{item}'].reshape(-1)  # [B*15*34]
#         keep = ~is_nan
#     else:
#         keep = slice(None)  # keep everything  # torch.zeros_like(true).bool()
#     if true is not None:
#         true = true[keep].cpu().numpy()
#     pred_mean = out[f'{item}'][keep, 0].cpu().numpy()  # [B*15*34]
#     pred_std = out[f'{item}'][keep, 1].cpu().numpy()**0.5  # [B*15*34]
#     if inverse_scale:
#         if item == 'distances':
#             if remove_nan:
#                 # Need not worry about masking since nans were removed
#                 if true is not None:
#                     true = inverse_scale_dist(true)
#                 pred_mean = inverse_scale_dist(pred_mean)
#                 pred_std = inverse_scale_dist_std(pred_std)
#             else:
#                 # Need to mask nan values before inverse transform
#                 if f'flag_{item}' in batch:
#                     nan_mask = batch[f'flag_{item}'].reshape(-1).cpu().numpy()
#                 else:
#                     nan_mask = np.zeros_like(true).astype(bool)
#                 # nan_mask ~ [B, 15, 34] --> [B*15*34]
#                 if true is not None:
#                     true[~nan_mask] = inverse_scale_dist(true[~nan_mask])
#                 pred_mean[~nan_mask] = inverse_scale_dist(pred_mean[~nan_mask])
#                 pred_std[~nan_mask] = inverse_scale_dist_std(pred_std[~nan_mask])
#         elif item in ['phi', 'psi']:
#             if remove_nan:
#                 if true is not None:
#                     true = inverse_scale_angles(true)
#                 pred_mean = inverse_scale_angles(pred_mean)
#                 pred_std = inverse_scale_angles_std(pred_std)
#             else:
#                 if f'flag_{item}' in batch:
#                     nan_mask = batch[f'flag_{item}'].reshape(-1).cpu().numpy()
#                 else:
#                     nan_mask = np.zeros_like(true).astype(bool)
#                 if true is not None:
#                     true[~nan_mask] = inverse_scale_angles(true[~nan_mask])
#                 pred_mean[~nan_mask] = inverse_scale_angles(pred_mean[~nan_mask])
#                 pred_std[~nan_mask] = inverse_scale_angles_std(pred_std[~nan_mask])
#     stats = {
#         'true': true,
#         'pred_mean': pred_mean,
#         'pred_std': pred_std,
#     }
#     return stats


def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode

    Note
    ----
    By Osvaldo Martin, from the book "Bayesian Analysis with Python"

    Parameters
    ----------
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results

    Returns
    ----------
    hpd: array with the lower

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
        x_hpd = x[(x > value[0]) & (x < value[1])]
        y_hpd = y[(x > value[0]) & (x < value[1])]
        modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    out_dict = {
        'hpd': hpd,
        'x': x,
        'y': y,
        'modes': modes,
        'density': density}
    return out_dict


def get_stats_from_samples(
    batch, out, item, inverse_scale_fn=lambda x: x, remove_nan=False):
    """
    Get inference results for the given batch

    Parameters
    ----------
    batch : dict
        Optionally contains f'{item'}, the label for the item
    out : dict
        Contains `dihedrals_samples' of shape `[num_samples, batch_size, 15, 2]`
    inverse_scale_fn : callable
        Function to inverse transform the quantities back to their physical units
    remove_nan : bool
        Whether to remove distance or angle elements that had nan labels

    """
    # Get label if available
    true = batch[f'{item}'].reshape(-1) if item in batch else None # [B*15*34,] for distances or [B*15,] for angles
    is_nan = batch[f'flag_{item}'].reshape(-1)  # [B*15*34,] for distances or [B*15,] for angles
    is_nan_np = is_nan.cpu().numpy()

    if remove_nan:
        keep = ~is_nan
    else:
        keep = slice(None)  # keep everything  # torch.zeros_like(true).bool()
    if true is not None:
        true[~is_nan] = inverse_scale_fn(true[~is_nan])
        true = true[keep].cpu().numpy()

    slice_idx = 0 if item in ['phi', 'distances'] else 1
    if item in ['phi', 'psi']:
        samples = out['dihedrals_samples'] # [N, B*15, 2]
    elif item == 'distances':
        samples = out['distances_samples'] # [N, B*15*34, 1]
    samples = inverse_scale_fn(samples)[:, :, slice_idx]  # [N, batch_size]

    out_dict = {
        'samples': samples,
        'true': true,  # [B*34*15,] or [B*15,]
        'pred_mean': samples.mean(0),  # [B*34*15,] or [B*15, ]
        'pred_std': samples.std(0),  # [B*34*15,] or [B*15, ]
    }
    return out_dict


# def get_stats_joint(batch, out, item, inverse_scale=False, remove_nan=False):
#     """
#     Get inference results for the given batch

#     Parameters
#     ----------
#     inverse_scale : bool
#         Whether to inverse transform the quantities back to their physical units
#     remove_nan : bool
#         Whether to remove distance or angle elements that had nan labels

#     """
#     # Get label if available
#     true = batch[f'{item}'].reshape(-1) if item in batch else None
#     if remove_nan:
#         is_nan = batch[f'flag_{item}'].reshape(-1)  # [B*15*34]
#         keep = ~is_nan
#     else:
#         keep = slice(None)  # keep everything  # torch.zeros_like(true).bool()
#     if true is not None:
#         true = true[keep].cpu().numpy()

#     if inverse_scale:
#         if item == 'distances':
#             # Pred mu, sigma for dist
#             pred_mean = out[f'{item}'][keep, 0].cpu().numpy()  # [B*15*34]
#             pred_std = out[f'{item}'][keep, 1].cpu().numpy()**0.5  # [B*15*34]
#             if remove_nan:
#                 # Need not worry about masking since nans were removed
#                 if true is not None:
#                     true = inverse_scale_dist(true)
#                 pred_mean = inverse_scale_dist(pred_mean)
#                 pred_std = inverse_scale_dist_std(pred_std)
#             else:
#                 # Need to mask nan values before inverse transform
#                 if f'flag_{item}' in batch:
#                     nan_mask = batch[f'flag_{item}'].reshape(-1).cpu().numpy()
#                 else:
#                     nan_mask = np.zeros_like(true).astype(bool)
#                 # nan_mask ~ [B, 15, 34] --> [B*15*34]
#                 if true is not None:
#                     true[~nan_mask] = inverse_scale_dist(true[~nan_mask])
#                 pred_mean[~nan_mask] = inverse_scale_dist(pred_mean[~nan_mask])
#                 pred_std[~nan_mask] = inverse_scale_dist_std(pred_std[~nan_mask])
#         elif item in ['phi', 'psi']:
#             # Pred mu, sigma from predictive samples for angles
#             # [B, n_samples, 2] --> [B, 2]
#             slice_idx = 0 if item == 'phi' else 1
#             pred_mean = out['dihedrals_samples'].mean(1)  # [B*15, 2]
#             pred_std = out['dihedrals_samples'].std(1)  # [B*15, 2]
#             if remove_nan:
#                 keep = keep.cpu().numpy()
#             pred_mean = pred_mean[keep, slice_idx]
#             pred_std = pred_std[keep, slice_idx]
#             if remove_nan:
#                 if true is not None:
#                     if item == 'phi':
#                         true = inverse_scale_phi(true)
#                     else:
#                         true = inverse_scale_psi(true)
#                 if item == 'phi':
#                     pred_mean = inverse_scale_phi(pred_mean)
#                 else:
#                     pred_mean = inverse_scale_psi(pred_mean)
#                 pred_std = inverse_scale_angles_std(pred_std)
#             else:
#                 if f'flag_{item}' in batch:
#                     nan_mask = batch[f'flag_{item}'].reshape(-1).cpu().numpy()
#                 else:
#                     nan_mask = np.zeros_like(true).astype(bool)
#                 if true is not None:
#                     if item == 'phi':
#                         true[~nan_mask] = inverse_scale_phi(true[~nan_mask])
#                     else:
#                         true[~nan_mask] = inverse_scale_psi(true[~nan_mask])
#                 if item == 'phi':
#                     pred_mean[~nan_mask] = inverse_scale_phi(pred_mean[~nan_mask])
#                 else:
#                     pred_mean[~nan_mask] = inverse_scale_psi(pred_mean[~nan_mask])
#                 pred_std[~nan_mask] = inverse_scale_angles_std(pred_std[~nan_mask])
#     stats = {
#         'true': true,
#         'pred_mean': pred_mean,
#         'pred_std': pred_std
#     }
#     return stats


def get_gmm_stats(batch, out, gmm_obj, item):
    """
    Parameters
    ----------
    batch : dict
        Required keys `{item}, flag_{item}`
    out : dict
        Required keys `dihedrals`
    gmm_obj
    item : str
        One of `[phi, psi]`

    """
    if item not in ['phi', 'psi', 'distances']:
        raise ValueError("`item` must be one of [phi, psi, distances]")

    # Get label (assumed available in `batch`)
    true = batch[f'{item}'].cpu().reshape(-1)  # [batch_size,] where batch_size = B*15
    # is_nan = batch[f'flag_{item}'].cpu()  # [batch_size, 15, 2]
    keep = ~torch.isnan(true)  # always remove nan
    true = true.unsqueeze(-1)  # [batch_size, 1]

    if item in ['phi', 'psi']:
        slice_idx = 0 if item == 'phi' else 1
        formatted = gmm_obj._format(out['dihedrals'].cpu().detach())
        _, (mix, comp) = gmm_obj._get_distribution(**formatted)
        probs = mix.probs  # [batch_size, num_components]
        mu = comp.loc[:, :, slice_idx]  # [batch_size, num_components]
        sigma = comp.covariance_matrix[:, :, slice_idx, slice_idx]**0.5  # [batch_size, num_components]
        # Modify truth to be the nearest angle
        true  = mu + torch.asin(
            torch.sin(
                (true - mu)*np.pi
                )
            )/np.pi  # asin output in [-pi/2, pi/2] so convert back to [-0.5, 0.5]
        # Modify mu to be the nearest angle
        # mu = true + torch.asin(
        #     torch.sin(
        #         (mu - true)*np.pi
        #     )
        #     )/np.pi # asin output in [-pi/2, pi/2] so convert back to [-0.5, 0.5]  
        # true = true.repeat(1, num_components)
    else:
        # Compute score for each component, then take weighted average
        formatted = gmm_obj._format(out['distances'].cpu().detach())
        _, (mix, comp) = gmm_obj._get_distribution(**formatted)
        probs = mix.probs  # [batch_size, num_components]
        mu = comp.loc[:, :, 0]  # [batch_size, num_components]
        sigma = comp.covariance_matrix[:, :, 0, 0]**0.5  # [batch_size, num_components]
        num_components = probs.shape[-1]
        true_eff = true.clone()  # [B*15*34,]
        true = true.repeat(1, num_components)  # [B*15*34, num_components]
        # Fit a single Gaussian on GMM samples
        samples = gmm_obj.sample(
            out['distances'].detach().cpu(),
            size=(100,)).squeeze(2)  # [num_samples, B*15*34]
        eff_mu = samples.mean(0)
        eff_sigma = samples.std(0)

    score = (mu - true)/sigma  # [batch_size, num_components]
    score = (score*probs).sum(1)  # [batch_size]

    # Convert to numpy
    stats = {
        "true": true.squeeze()[keep].numpy(),  # [batch_size,]
        "score": score[keep].numpy(),  # [batch_size,]
        "probs": probs[keep].numpy(),  # [batch_size, num_components]
        "pred_mean": mu[keep].numpy(),  # [batch_size, num_components]
        "pred_std": sigma[keep].numpy(),  # [batch_size, num_components],
        "flags": batch[f'flag_{item}'].cpu().numpy(),
        "peptide_len": batch['peptide_len'].cpu().numpy().astype(int),
    }
    if item == 'distances':
        eff_score = (eff_mu - true_eff)/eff_sigma  # [batch_size,]
        stats.update(
            eff_true=true_eff.squeeze(1).numpy(),
            eff_score=eff_score[keep].numpy(),  # [batch_size,]
            eff_mu=eff_mu.numpy(),  # [batch_size,]
            eff_sigma=eff_sigma.numpy(),  # [batch_size,]
            )
    return stats


def plot_gmm(true, pred_mean, score, item, inverse_scale=True, to_PIL=True, scaling='scalar', **kwargs):
    """

    Parameters
    ----------
    true : np.ndarray
        [batch_size, num_components]
    score : np.ndarray
        [batch_size,]
    pred_mean : np.ndarray
        [batch_size, num_components]

    """
    plt.close('all')
    B, num_components = pred_mean.shape
    fig, ax = plt.subplots()
    if item == 'phi':
        inverse_scale_fn = inverse_scale_phi
        # if 'flip' in scaling:
        #     inverse_scale_fn = inverse_scale_angles_flip
    elif item == 'psi':
        inverse_scale_fn = inverse_scale_psi
        # if 'flip' in scaling:
        #     inverse_scale_fn = inverse_scale_angles_flip
    elif item == 'distances':
        if scaling == 'scalar':
            inverse_scale_fn = inverse_scale_dist
        elif 'log' in scaling:
            inverse_scale_fn = inverse_scale_dist_log
    else:
        raise ValueError("Must be one of `[phi, psi, distances]`")
    if inverse_scale:
        true = inverse_scale_fn(true)
        pred_mean = inverse_scale_fn(pred_mean)
    # plot sin instead
    # if item == 'phi' or item == 'psi':
    #     true = torch.sin(torch.tensor(true)/180*np.pi).numpy()
    #     pred_mean = torch.sin(torch.tensor(pred_mean)/180*np.pi).numpy()
    for comp in range(num_components):
        points = ax.scatter(
            true[:, comp], pred_mean[:, comp], c=score, marker='.', alpha=0.3) #true[:, comp]
    cbar = fig.colorbar(points)  # same colorbar
    # cbar.set_label('Number of predicted STD away from truth')

    if item in ['phi', 'psi']:
        grid = np.linspace(-180 - 90.5, 180 + 90.5, 20)
        ax.set_xlabel('True / deg')
        ax.set_ylabel('Pred mean / deg')
        ax.set_xlim([-180 - 90.5, 180 + 90.5])
        ax.set_ylim([-180 - 90.5, 180 + 90.5])
    else:
        grid = np.linspace(0.0, 35.0, 20) #35
        ax.set_xlabel('True / Ang')
        ax.set_ylabel('Pred mean / Ang')
        ax.set_xlim([0.0, 35.0]) #35
        ax.set_ylim([0.0, 35.0])
    ax.plot(grid, grid, color='tab:gray', linestyle='--')
    if to_PIL:
        return buffer_plot_and_get(fig)
    else:
        return fig


def plot_per_position(
        eff_mu, eff_sigma, eff_true, flags, peptide_len,
        item, inverse_scale=True, to_PIL=True, scaling='scalar', **kwargs):
    if item != 'distances':
        raise NotImplementedError(
            "Per-pos plotting only implemented for distances")
    is_nan = flags.reshape(-1, 15, 34)
    labels =  eff_true.reshape(-1, 15, 34)
    pred_mean = eff_mu.reshape(-1, 15, 34)
    pred_std = eff_sigma.reshape(-1, 15, 34)

    batch_size = labels.shape[0]
    if inverse_scale:
        if scaling == 'scalar':
            labels = inverse_scale_dist(labels)
            pred_mean = inverse_scale_dist(pred_mean)
            pred_std = inverse_scale_dist_std(pred_std)
        elif 'log' in scaling:
            labels = inverse_scale_dist_log(labels)
            pred_mean = inverse_scale_dist_log(pred_mean)
            pred_std = inverse_scale_dist_std_log(pred_std)

    # Compute some metrics
    abs_bias = np.abs(pred_mean - labels)
    bias = (pred_mean - labels)
    score = (pred_mean - labels)/pred_std

    # Store in df for convenient access
    pep_len_df = pd.DataFrame()
    pep_len_df['Peptide length'] = np.tile(
        peptide_len[:, np.newaxis, np.newaxis],
        (1, 15, 34)).reshape(-1)
    pep_len_df['Position'] = np.tile(
        np.arange(1, 16)[np.newaxis, :, np.newaxis],
        (batch_size, 1, 34)).reshape(-1)
    pep_len_df['Bias (Å)'] = bias.reshape(-1)
    pep_len_df['Sigma'] = pred_std.reshape(-1)
    pep_len_df['Abs. Bias'] = abs_bias.reshape(-1)
    pep_len_df['Score'] = score.reshape(-1)
    pep_len_df['remove'] = is_nan.reshape(-1)
    pep_len_df = pep_len_df[~pep_len_df['remove']]

    plt.close('all')
    fig, ax = plt.subplots(3, figsize=(18, 4*3))
    for i, item in enumerate(['Bias (Å)', 'Abs. Bias', 'Score']):
        sns.boxplot(data=pep_len_df, x="Position", y=item, hue='Peptide length', ax=ax[i])
        sns.move_legend(ax[i], "upper left", bbox_to_anchor=(1, 1))
        ax[i].axhline(0.0, color='tab:gray', linestyle='--')
        if item == 'Score':
            ax[i].axhline(3.0, color='tab:gray', linestyle='--')
            ax[i].axhline(-3.0, color='tab:gray', linestyle='--')
    return fig

def save_dist_angles_perres(inname, peptide_lens, save_path, ind, outname):
    out = np.load(os.path.join(save_path, inname), allow_pickle=True)
    dict_out = out.item()
    fig_dist = plot_recovery_dist(
            true=dict_out['distances_labels'],
            pred_mean=dict_out['distances_mean'],
            pred_std=dict_out['distances_std'], to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_dist')):
        os.mkdir(os.path.join(save_path, 'recovery_dist'))
    fig_dist.savefig(os.path.join(save_path, f'recovery_dist/{outname}'),dpi=1200)

    fig_phi = plot_recovery_angles(
                true=dict_out['phi_true'],
                pred_mean=dict_out['phi_mean'],
                pred_std=dict_out['phi_std'],
                inverse_scale = False,
                to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_phi')):
        os.mkdir(os.path.join(save_path, 'recovery_phi'))
    fig_phi.savefig(os.path.join(save_path, f'recovery_phi/{outname}'),dpi=1200)

    fig_psi = plot_recovery_angles(
                true=dict_out['psi_true'],
                pred_mean=dict_out['psi_mean'],
                pred_std=dict_out['psi_std'],
                inverse_scale=False,
                to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_psi')):
        os.mkdir(os.path.join(save_path,'recovery_psi'))
    fig_psi.savefig(os.path.join(save_path, f'recovery_psi/{outname}'),dpi=1200)

    fig_psi_flip = plot_recovery_angles(
                true=dict_out['psi_wraptrue'],
                pred_mean=dict_out['psi_mean'],
                pred_std=dict_out['psi_std'],
                inverse_scale=False,
                to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_psi_wrap')):
        os.mkdir(os.path.join(save_path,'recovery_psi_wrap'))
    fig_psi_flip.savefig(os.path.join(save_path, f'recovery_psi_wrap/{outname}'),dpi=1200)

    fig_phi_flip = plot_recovery_angles(
                true=dict_out['phi_wraptrue'],
                pred_mean=dict_out['phi_mean'],
                pred_std=dict_out['phi_std'],
                inverse_scale=False,
                to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_phi_wrap')):
        os.mkdir(os.path.join(save_path,'recovery_phi_wrap'))
    fig_phi_flip.savefig(os.path.join(save_path, f'recovery_phi_wrap/{outname}'),dpi=1200)

    fig_per_residue = plot_per_position(
                        dict_out['distances_mean'], 
                        dict_out['distances_std'],
                        dict_out['distances_labels'], 
                        dict_out['distances_flags'], 
                        peptide_lens,
                        'distances', inverse_scale=False, to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_perresidue')):
        os.mkdir(os.path.join(save_path,'recovery_perresidue'))
    fig_per_residue.savefig(os.path.join(save_path, f'recovery_perresidue/{outname}'), dpi=1200)

def inverse_scale_true(item,true_df):
    true = np.vstack(true_df[f'scaled_{item}']).reshape(-1)
    is_nan = np.vstack(true_df[f'flag_{item}']).reshape(-1)  # [B*15*34] or [B*15]
    if item == 'phi':
        inverse_scale_fn = inverse_scale_phi
    elif item == 'psi':
        inverse_scale_fn = inverse_scale_psi
    true[~is_nan] = inverse_scale_fn(true[~is_nan])
    return true

def plot_dihedrals_with_specified_scaling(inname, true_df, save_path, ind, outname):
    out = np.load(os.path.join(save_path, inname), allow_pickle=True)
    dict_out = out.item()
    samples = dict_out['dihedrals_samples'].reshape(1000, -1, 2).copy()
    samples_phi = inverse_scale_phi(samples)[:, :, 0]
    samples_psi = inverse_scale_psi(samples)[:, :, 1]
    phi_mean = samples_phi.mean(0)
    psi_mean = samples_psi.mean(0)
    phi_std = samples_phi.std(0)
    psi_std = samples_psi.std(0)
    true_phi = inverse_scale_true('phi', true_df)
    true_psi = inverse_scale_true('psi', true_df)        
    fig_phi = plot_recovery_angles(
            true=true_phi,
            pred_mean=phi_mean,
            pred_std=phi_std,
            inverse_scale = False,
            to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_phi')):
        os.mkdir(os.path.join(save_path,'recovery_phi'))
    fig_phi.savefig(os.path.join(save_path, f'recovery_phi/{outname}'), dpi=1200)
    
    fig_psi = plot_recovery_angles(
            true=true_psi,
            pred_mean=psi_mean,
            pred_std=psi_std,
            inverse_scale = False,
            to_PIL=False)
    if not os.path.isdir(os.path.join(save_path, 'recovery_psi')):
        os.mkdir(os.path.join(save_path,'recovery_psi'))
    fig_psi.savefig(os.path.join(save_path, f'recovery_psi/{outname}'), dpi=1200)