import math
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC
import esm
# from philia.models.layers import StochasticVariationalGP, CustomVariationalELBO
# from philia.models.nll import FullRankGaussianNLL
# from philia.models.interface_model import InterfaceModel
from philia.analysis.plotting import (
    # plot_recovery_dist, plot_recovery_angles,
    plot_gmm,
    # get_stats,
    get_stats_from_samples,
    get_gmm_stats,
    plot_per_position)
    #plot_hpd_dist, plot_hpd_angles)
from philia.trainval_data.transforms import (
    inverse_scale_dist,
    inverse_scale_dist_log,
    inverse_scale_angles,
    # inverse_scale_angles_flip,
    inverse_scale_phi,
    inverse_scale_psi,
    inverse_scale_dist_std,
    inverse_scale_dist_std_log)
from mm import FullRankGMM
import wandb
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 320, max_len: int = 34):
        # dropout: float = 0.0,
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[[0], :x.size(1)]
        return x  # self.dropout(x)


class JointInterfaceModel(pl.LightningModule): #InterfaceModel
    def __init__(self,
                 lr=None, wd=None,
                 interface_pred=False, freeze_embeddings=True,
                #  use_gp=False,
                #  dist_only=False,
                 lr_patience=8,
                 d_mlp=64,
                 weight_dihedral_loss=0.02,
                 weight_aug=1.0,
                 weight_length=0.6,
                 num_components=1,
                 scale_option='scalar'):
        """

        Parameters
        ----------
        simple_mlp : bool
            Whether to have a simple (single FC) regression MLP. Default: True
            (defaults to R2 setting)
        use_layernorm : bool
            Whether to use layernorm after embedding layer. Default: False
            (defaults to R2 setting)

        """
        super().__init__()
        self.save_hyperparameters()
        self.wd = wd
        self.lr = lr
        self.interface_pred = interface_pred
        self.freeze_embeddings = freeze_embeddings
        # self.use_gp = use_gp
        # self.dist_only = dist_only
        self.scale_option = scale_option
        # if dist_only:
        #     raise ValueError("Use InterfaceModel instead.")
        self.lr_patience = lr_patience
        self.esm_model_hla, self.alphabet_hla = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model_peptide, self.alphabet_peptide = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter_hla = self.alphabet_hla.get_batch_converter()
        self.batch_converter_peptide = self.alphabet_peptide.get_batch_converter()
        self.final_layer = 6
        self.d_embed = 320
        self.d_mlp = d_mlp
        self.weight_dihedral_loss = weight_dihedral_loss
        self.weight_aug = weight_aug
        self.weight_length = weight_length
        self.num_components = num_components
        self.pos_encoder = PositionalEncoding(self.d_embed, max_len=34)
        self.mlp = nn.Sequential(nn.Linear(self.d_embed*2, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))
        if self.interface_pred and self.freeze_embeddings:
            for param in self.esm_model_hla.parameters():
                param.requires_grad = False
            for param in self.esm_model_peptide.parameters():
                param.requires_grad = False
            for param in self.mlp.parameters():
                param.requires_grad = False
        self.layernorm = nn.LayerNorm(self.d_embed)
        self.layernorm_pep = nn.LayerNorm(self.d_embed)
        self.layernorm_hla = nn.LayerNorm(self.d_embed)
        # Interface predictor GP
        if self.interface_pred:
            self.d_embed_interface = 32
            self.mlp_pep = nn.Sequential(
                nn.Linear(self.d_embed, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_embed_interface))
            self.mlp_hla = nn.Sequential(
                nn.Linear(self.d_embed, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_embed_interface))
            # if self.use_gp:
            #     raise NotImplementedError(
            #         "GP is currently only supported in non-joint version.")
            # else:
                # self.nll_dihedrals = FullRankGaussianNLL(
                #     2,  # phi, psi
                #     device=torch.device('cpu')  # to replace with input device
                # )
            self.nll_dihedrals = FullRankGMM(
                dim=2,  # phi, psi
                num_components=self.num_components,
                is_circular=True)
            self.nll_dist = FullRankGMM(
                dim=1, num_components=self.num_components)
            # For Y_dim = 2,
            # d_out = mu (2) + L (3) for dihedrals + mu, sigma (2) for dist
            self.d_out = self.nll_dihedrals.out_dim + 2
            self.dist_idx = [0, 1]  # mu, sigma
            self.dihedrals_idx = torch.arange(self.d_out)[2:]  # [2, ...]
            self.dihedrals_cov_idx = torch.arange(
                self.d_out)[self.nll_dihedrals.tril_len:]
            self.mlp_dihedrals = nn.Sequential(
                nn.Linear(self.d_embed_interface, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.nll_dihedrals.out_dim))
            self.mlp_dist = nn.Sequential(
                nn.Linear(self.d_embed_interface, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.d_mlp),
                nn.ReLU(),
                nn.Linear(self.d_mlp, self.nll_dist.out_dim))

        self.auroc = AUROC(task='binary')
        self.validation_step_outputs = []

    def forward(self, batch):
        batch_size = batch['peptide_len'].shape[0]
        hla = self.esm_model_hla(
            batch['hla_sequence'],
            repr_layers=[self.final_layer])  # [B, 34+2, 320]
        pep = self.esm_model_peptide(
            batch['peptide'],
            repr_layers=[self.final_layer])  # [B, 15+2, 320]
        hla = hla['representations'][self.final_layer]  # [B, 34+2, 320]
        pep = pep['representations'][self.final_layer]  # [B, 15+2, 320]
        hla = hla[:, 1:-1, :]  # [B, 34, 320]
        pep = pep[:, 1:-1, :]  # [B, 15, 320]
        hla = self.layernorm_hla(hla)  # [B, 34, 320]
        pep = self.layernorm_pep(pep)  # [B, 15, 320]
        hla = self.pos_encoder(hla)  # [B, 34, 320]
        pep = self.pos_encoder(pep)  # [B, 15, 320]
        if not self.interface_pred:  # binary classifier training
            pep_representations = []
            for i, pep_len in enumerate(batch['peptide_len']):
                pep_representations.append(
                    pep[i, :pep_len].mean(0))  # [pep_len, 320] --> [320,]
            pep = torch.stack(pep_representations)  # [B, 320]
            hla = hla.mean(1)  # [B, 34, 320] --> [B, 320]
            shared = torch.cat([hla, pep], dim=1)  # [B, 640]
            y = self.mlp(shared)  # [B, 1]
            return y
        else:  # interface constraints pred
            pep_latent = pep.reshape(-1, self.d_embed)  # [B*15, 320]
            pep_latent = self.mlp_pep(pep_latent)  # [B*15, d_embed_interface]
            hla_latent = hla.reshape(-1, self.d_embed)  # [B*34, 320]
            hla_latent = self.mlp_hla(hla_latent)  # [B*34, d_embed_interface]
            # hla_latent = self.layernorm_hla(hla_latent)
            # Take outer product to compute every pair --> [B*15*34, d_embed_interface]
            outer_latent = torch.einsum(
                'bik,bjk->bijk',
                pep_latent.reshape(batch_size, 15, self.d_embed_interface),
                hla_latent.reshape(batch_size, 34, self.d_embed_interface))  # [B, 15, 34, d_embed_interface]
            outer_latent_dist = outer_latent.reshape(-1, self.d_embed_interface)  # [B*15*34, d_embed_interface]
            outer_latent_angles = outer_latent.mean(2).reshape(-1, self.d_embed_interface)  # [B*15, d_embed_interface]
            out_dict = {}
            # if self.use_gp:
            #     raise NotImplementedError(
            #         "GP is currently only supported in non-joint version.")
            # else:
            # if not self.dist_only:
            out_dihedrals = self.mlp_dihedrals(
                outer_latent_angles)  # [B*15, out_dim], out_dim=18 if num_components=3
            out_dihedrals = out_dihedrals.reshape(
                batch_size*15, self.nll_dihedrals.num_components, -1
            )
            # Force mu to lie in [-1, 1]
            mu = torch.sigmoid(out_dihedrals[:, :, [1, 2]])*2.0 - 1.0
            # [batch_size, num_components, 2]
            # mu = torch.cat(
            #     [
            #         mu[:, :, [0]] + 0.5,  # [-0.5, 1.5]
            #         mu[:, :, [1]] - 0.75,  # [-1.75, 0.25]
            #     ], dim = -1
            # )  # still [batch_size, num_components, 2]
            # mu = torch.cat(
            #     [
            #         mu[:, :, [0]],
            #         mu[:, :, [1]],
            #     ], dim = -1
            # )
            out_dihedrals = torch.cat(
                [
                    out_dihedrals[:, :, [0]],
                    mu,
                    out_dihedrals[:, :, 3:],
                ], dim=-1)
            out_dihedrals = out_dihedrals.reshape(-1, self.nll_dihedrals.out_dim) # [B*15, out_dim]
            out_dict.update(dihedrals=out_dihedrals)
            out_distance = self.mlp_dist(outer_latent_dist) # [B*15*34, out_dim], out_dim=9 if num_components=3
            # Modify mu
            # out_distance = out_distance.reshape(
            #     batch_size*15*34, self.nll_dist.num_components, -1)
            # mu = torch.sigmoid(out_distance[:, :, [1]])*2.0 - 1.0
            # out_distance = torch.cat(
            #     [
            #         out_distance[:, :, [0]],
            #         mu,
            #         out_distance[:, :, 2:]
            #     ], dim=-1
            # )
            # out_distance = out_distance.reshape(-1, self.nll_dist.out_dim)
            # Modify sigma
            # out_distance = out_distance.reshape(
            #         batch_size*15*34, self.nll_dist.num_components, -1) # [B*15*34, num_components, -1]
            # pos_std_dims = [
            #         out_distance[:, :, [0, 1]],
            # #     # sigma must be pos
            #         torch.sigmoid(out_distance[:, :, [2]])*2.0 - 1.0,
            # #     # F.softplus(out_distance[:, :, [2]], beta=2.0),
            # #     # out_distance[:, :, [2]] + 1.0,
            #         ]
            # out_distance = torch.cat(pos_std_dims, dim=-1)  # [B*15*34, num_components, -1]
            # out_distance = out_distance.reshape(-1, self.nll_dist.out_dim) # [B*15*34, out_dim]
            # End modify sigma
            out_dict.update(distances=out_distance)
            return out_dict

    def configure_optimizers(self):
        params_list = [{'params': self.parameters()}]
        # if self.interface_pred:
        #     params_list.append({'params': self.phi_gp.likelihood.parameters()})
        #     params_list.append({'params': self.psi_gp.likelihood.parameters()})
        #     params_list.append({'params': self.distance_gp.likelihood.parameters()})
        optimizer = torch.optim.Adam(
            params_list,
            lr=self.lr,
            weight_decay=self.wd)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.lr_patience,
            min_lr=1.e-5, verbose=True,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss_early_stopping'}

    def training_step(self, train_batch, batch_idx):
        out = self(train_batch)
        if not self.interface_pred:
            # loss = F.mse_loss(self(x), y)
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(-1), train_batch['label'],
                weight=train_batch['loss_weight'])
            wandb.log({'train_loss': loss.detach().cpu()})
        else:
            loss = self.compute_nll_loss(train_batch, out)
            self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        if not self.interface_pred:
            self.validation_step_classifier(val_batch, batch_idx)
        else:
            self.validation_step_regressor(val_batch, batch_idx)

    def validation_step_classifier(self, val_batch, batch_idx):
        out = self(val_batch)
        # loss = F.mse_loss(self(x), y)
        loss = F.binary_cross_entropy_with_logits(
            out.squeeze(-1), val_batch['label'])
        self.log('val_loss_early_stopping', loss, on_step=True, logger=True)
        sch = self.lr_schedulers()
        sch.step(loss)
        pred_probs = torch.sigmoid(out)
        wandb.log(
            {
                'auroc': self.auroc(
                    pred_probs.squeeze(-1), val_batch['label'].long()),
                'val_loss': loss
            })
        pred_probs = pred_probs.cpu().numpy()  # [B,]
        pred_probs_2d = np.concatenate([1.0-pred_probs, pred_probs], axis=-1)  # [B, 2]
        labels = val_batch['label'].cpu().numpy()  # [B,]
        # import pdb; pdb.set_trace()
        wandb.log({"roc": wandb.plot.roc_curve(
            labels, pred_probs_2d)})
        wandb.log({"prc": wandb.plot.pr_curve(
            labels, pred_probs_2d)})
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(
            y_true=labels, probs=pred_probs_2d,
            class_names=['no-bind', 'bind'])})

    def compute_nll_loss(self, batch, out, is_train=True):
        #############
        # Distances #
        #############
        # batch['flag_distances'] ~ [B, 15, 34]
        # batch['distances'] ~[B, 15, 34]
        batch_size, max_pep_len, _ = batch['flag_distances'].shape
        is_nan = batch['flag_distances'].reshape(-1)  # [B, 15, 34] --> [B*15*34,]
        batch_size_eff = batch_size*max_pep_len  # B*15

        dist_labels = batch['distances'].reshape(-1)[~is_nan]
        if dist_labels.ndim == 1:
            dist_labels = dist_labels.unsqueeze(-1)
        loss_dist = self.nll_dist(
            out['distances'][~is_nan],
            label=dist_labels)
            # out['distances'][~is_nan, 0],
            # batch['distances'].reshape(-1)[~is_nan],
            # out['distances'][~is_nan, 1])
        # For upweighting xtal & downweighting 9mer
        if is_train:
            dist_weights = batch['is_xtal_distances'].reshape(-1)  # [B*15*34,]
            dist_weights = torch.clamp(dist_weights, min=self.weight_aug)
            dist_weights = dist_weights * torch.clamp((batch['is_9mer_distances'].reshape(-1)==0).to(torch.float), min=self.weight_length)
            loss_dist = loss_dist * dist_weights[~is_nan]

        loss_dist = loss_dist.mean()
        # [(torch.rand(loss_dist.shape[0]) > 0.5).bool().to(loss_dist.device)]
        loss = loss_dist.mean()

        #############
        # Dihedrals #
        #############
        # batch['flag_phi'] ~ [B, 15]
        # batch['flag_psi] ~ [B, 15]
        # batch['phi'] ~ [B, 15]
        # batch['psi'] ~ [B, 15]
        is_nan_dihedral = torch.stack(
            [batch['flag_phi'], batch['flag_psi']], dim=-1)  # [B, 15, 2]
        is_nan_dihedral = is_nan_dihedral.reshape(-1, 2)  # [B*15, 2]
        both_observed = torch.all(
            torch.eq(
                is_nan_dihedral,
                torch.tensor([False, False]).to(is_nan_dihedral.device)), dim=1)
        phi_observed = torch.all(
            torch.eq(
                is_nan_dihedral,
                torch.tensor([False, True]).to(is_nan_dihedral.device)), dim=1)
        psi_observed = torch.all(
            torch.eq(
                is_nan_dihedral,
                torch.tensor([True, False]).to(is_nan_dihedral.device)), dim=1)

        # For upweighting xtal
        if is_train:
            phi_weights = torch.clamp(
                batch['is_xtal_phi'].reshape(-1), min=self.weight_aug)   # [B*15]
            psi_weights = torch.clamp(
                batch['is_xtal_psi'].reshape(-1), min=self.weight_aug)   # [B*15]
        
        # For downweighting 9mer
        if is_train:
            phi_weights = phi_weights * torch.clamp(
                (batch['is_9mer_phi'].reshape(-1)==0).to(torch.float), min=self.weight_length) # [B*15] 
            psi_weights = psi_weights * torch.clamp(
                (batch['is_9mer_psi'].reshape(-1)==0).to(torch.float), min=self.weight_length)

        # batch['phi] ~ [B, 15]
        # batch['psi] ~ [B, 15]
        # self.nll_dihedrals.set_device(is_nan_dihedral.device)
        labels_dihedrals = torch.stack([batch['phi'], batch['psi']], dim=-1)  # [B, 15, 2]
        labels_dihedrals = labels_dihedrals.reshape(batch_size_eff, 2).float()  # [B*15, 2]
        labels_dihedrals = labels_dihedrals.unsqueeze(1)  # [B*15, 1, 2]
        out_dihedrals = out['dihedrals'] # [B, num_components*]
        mu = out_dihedrals.reshape(
            batch_size_eff, self.nll_dihedrals.num_components, -1)[:, :, 1:3].detach()
        # flip true
        diff = torch.asin(
            torch.clamp(torch.sin((labels_dihedrals - mu)*np.pi),
                        min=-1.0 + 1.e-5,
                        max=1.0 - 1.e-5)
            )/np.pi  # asin output in [-pi/2, pi/2] so convert back to [-0.5, 0.5]
        # with torch.no_grad():
        #     already_nan = torch.isnan(labels_dihedrals.repeat(1, 3, 1))
        #     after_nan = torch.isnan(diff)
        #     print(torch.logical_and(~already_nan, after_nan).sum(), "new nans")
        labels_dihedrals = mu + diff

        # Iterate the sampling procedure for phi given psi distribution to make sure the final sample size is nonzero
        iter = 0
        while True:
            phi_given_psi_random = (
                    torch.rand(batch_size_eff) > 0.5).to(both_observed.device)
            phi_given_psi = torch.logical_and(
                phi_given_psi_random, both_observed)  # [B*15,]
            psi_given_phi = torch.logical_and(
                ~phi_given_psi_random, both_observed)  # [B*15,]
            if phi_given_psi.float().mean() > 0 and psi_given_phi.float().mean() > 0:
                print('Sampling finished at {}th iteration'.format(iter))
                break
            else:
                iter += 1
        phi_idx = 0
        psi_idx = 1

        # (neg) log p(phi|psi) + log p(psi)
        loss_phi_given_psi = self.nll_dihedrals.get_conditional_nll(
            out_pred=out_dihedrals[phi_given_psi],
            label=labels_dihedrals[phi_given_psi][..., [phi_idx]],
            condition_vals=labels_dihedrals[phi_given_psi][..., [psi_idx]],
            condition_idx=[psi_idx],
        )
        loss_psi = self.nll_dihedrals.get_marginal_nll(
            out_pred=out_dihedrals[torch.logical_or(psi_observed, phi_given_psi)],
            label=labels_dihedrals[
                torch.logical_or(psi_observed, phi_given_psi)][..., [psi_idx]],
            keep_idx=[psi_idx],
        )
        if is_train:
            loss_phi_given_psi = loss_phi_given_psi*phi_weights[phi_given_psi]
            loss_psi = loss_psi*psi_weights[torch.logical_or(psi_observed, phi_given_psi)]

        # (neg) log p(psi|phi) + log p(phi)
        loss_psi_given_phi = self.nll_dihedrals.get_conditional_nll(
            out_pred=out_dihedrals[psi_given_phi],
            label=labels_dihedrals[psi_given_phi][..., [psi_idx]],
            condition_vals=labels_dihedrals[psi_given_phi][..., [phi_idx]],
            condition_idx=[phi_idx],
        )

        loss_phi = self.nll_dihedrals.get_marginal_nll(
            out_pred=out_dihedrals[torch.logical_or(phi_observed, psi_given_phi)],
            label=labels_dihedrals[
                torch.logical_or(phi_observed, psi_given_phi)][..., [phi_idx]],
            keep_idx=[phi_idx],
        )

        weight_dihedral = 1
        if is_train:
            loss_psi_given_phi = loss_psi_given_phi*phi_weights[psi_given_phi]
            loss_phi = loss_phi*phi_weights[torch.logical_or(phi_observed, psi_given_phi)]
            weight_dihedral = self.weight_dihedral_loss
        
        loss_dihedrals = (
            loss_phi_given_psi.mean() + loss_psi.mean() + loss_psi_given_phi.mean() + loss_phi.mean()
        )*weight_dihedral  # arbitrary scaling relative to dist loss
        loss += loss_dihedrals
        self.log_dict({
            'loss': loss,
            'loss_dist': loss_dist.mean(),
            'loss_dihedrals': loss_dihedrals,})

        # wandb.log(
        #     # 'loss_phi_given_psi': wandb.Histogram(loss_phi_given_psi.detach().cpu().numpy()),
        #     # 'loss_psi': wandb.Histogram(loss_psi.detach().cpu().numpy()),
        #     # 'loss_psi_given_phi': wandb.Histogram(loss_psi_given_phi.detach().cpu().numpy()),
        #     # 'loss_phi': wandb.Histogram(loss_phi.detach().cpu().numpy()),
        #     # 'dihedrals loss': wandb.Histogram(loss_dihedrals.detach().cpu().numpy()),
        # })
        # wandb.log({'dist loss': wandb.Histogram(loss_dist.detach().cpu().numpy())})
        # logger.debug(f'Loss before reduction: {loss_dist}')
        return loss

    def validation_step_regressor(self, val_batch, batch_idx):
        out = self(val_batch)
        loss = self.compute_nll_loss(val_batch, out, is_train=False)
        log_dict = {}
        # self.log('val_loss', loss)

        # Plotting
        if batch_idx == 0:
            # Distances
            # dist_stats = get_stats(
            #     val_batch, out, 'distances', inverse_scale=True, remove_nan=True)
            dist_stats = get_gmm_stats(
                val_batch, out, self.nll_dist, 'distances')
            log_dict['distances'] = dist_stats

            # Angles
            # samples = self.nll_dihedrals.sample(
            #     out['dihedrals'], size=torch.Size([500,])).cpu()
            # out['dihedrals_samples'] = samples
            phi_stats = get_gmm_stats(
                val_batch, out, self.nll_dihedrals, 'phi')
            log_dict['phi'] = phi_stats
            psi_stats = get_gmm_stats(
                val_batch, out, self.nll_dihedrals, 'psi')
            log_dict['psi'] = psi_stats
            # wandb.log({'val_loss_batch0': loss,})
        log_dict['loss'] = loss
        self.validation_step_outputs.append(log_dict)
        return loss

    def on_validation_epoch_end(self):
        if not self.interface_pred:
            return
        epoch_loss = torch.stack(
            [d['loss'] for d in self.validation_step_outputs]).mean()
        log_dict_first_batch = self.validation_step_outputs[0]
        # Plot
        fig_dist = plot_gmm(**log_dict_first_batch['distances'], item='distances',scaling=self.scale_option)
        fig_phi = plot_gmm(**log_dict_first_batch['phi'], item='phi')
        fig_psi = plot_gmm(**log_dict_first_batch['psi'], item='psi')
        fig_dist_per_pos = plot_per_position(
            **log_dict_first_batch['distances'], item='distances',scaling=self.scale_option)
        wandb.log({
            'scores_dist': wandb.Histogram(log_dict_first_batch['distances']['score']),
            'scores_phi': wandb.Histogram(log_dict_first_batch['phi']['score']),
            'scores_psi': wandb.Histogram(log_dict_first_batch['psi']['score']),
            'eff_scores_dist': wandb.Histogram(log_dict_first_batch['distances']['eff_score']),
        })
        if self.scale_option == 'scalar':
            inverse_scale_dist_opt = inverse_scale_dist_std
        elif 'log' in self.scale_option:
            inverse_scale_dist_opt = inverse_scale_dist_std_log
        wandb.log({
            'std_dist': wandb.Histogram(
                inverse_scale_dist_opt(log_dict_first_batch['distances']['pred_std'])),
            # 'scores_phi': wandb.Histogram(log_dict_first_batch['phi']['score']),
            # 'scores_psi': wandb.Histogram(log_dict_first_batch['psi']['score']),
        })
        wandb.log({
            'val_dist_recovery': wandb.Image(fig_dist),
            'val_phi_recovery': wandb.Image(fig_phi),
            'val_psi_recovery': wandb.Image(fig_psi),
            'val_dist_per_pos': wandb.Image(fig_dist_per_pos)})
        self.log(
            'val_loss_early_stopping', epoch_loss,
            on_step=False, on_epoch=True, logger=True)
        sch = self.lr_schedulers()
        sch.step(epoch_loss)
        self.validation_step_outputs = []  # reset for next epoch

    def predict_step(self, batch, batch_idx):
        if not self.interface_pred:
            return self.get_classifier_preds(batch)
        else:
            out = self(batch)
            # Distances
            dist_samples = self.nll_dist.sample(
                out['distances'], size=(1000,)).cpu() # [1000, B*15*34, 1]
            out['distances_samples'] = dist_samples
            if self.scale_option == 'scalar':
                inverse_scale_dist_opt = inverse_scale_dist
            elif 'log' in self.scale_option:
                inverse_scale_dist_opt = inverse_scale_dist_log
            dist_stats = get_stats_from_samples(
                batch, out, 'distances',
                inverse_scale_fn=inverse_scale_dist_opt,
                remove_nan=False)
            # true, pred_mean, pred_std = get_stats_from_samples(
            #     batch, out, 'distances',
            #     inverse_scale_fn=inverse_scale_dist, remove_nan=False)
            out['distances_mean'] = dist_stats['pred_mean']
            out['distances_std'] = dist_stats['pred_std']
            # out['distances_true'] = dist_stats['true']
            if 'flag_distances' in batch:
                out.update(
                    distances_labels=dist_stats['true'],
                    distances_flags=batch['flag_distances'].cpu().numpy(),
                    distances_samples=dist_stats['samples'].cpu().numpy())

            # Angles
            samples = self.nll_dihedrals.sample(
                out['dihedrals'], size=torch.Size([1000,])).cpu() # [1000, B*15, 2]
            out['dihedrals_samples'] = samples
            for angle_type in ['phi', 'psi']:
                if angle_type == 'phi':
                    inverse_scale_fn = inverse_scale_phi
                    stats = get_stats_from_samples(
                        batch, out, angle_type,
                        inverse_scale_fn=inverse_scale_fn,
                        remove_nan=False)
                if angle_type == 'psi':
                    inverse_scale_fn = inverse_scale_psi
                    stats = get_stats_from_samples(
                        batch, out, angle_type,
                        inverse_scale_fn=inverse_scale_fn,
                        remove_nan=False)
                # wrap around the dihedral angles
                # stats pred_mean and true already inversely scaled
                diff = torch.asin(
                torch.clamp(torch.sin((torch.tensor(stats['true']) - stats['pred_mean'])/180*np.pi),
                            min=-1.0 + 1.e-5,
                            max=1.0 - 1.e-5)
                ).numpy()/np.pi*180  # asin output in [-pi/2, pi/2] so convert back to [-0.5, 0.5]
                labels_dihedrals = stats['pred_mean'] + diff

                out[f'{angle_type}_mean'] = stats['pred_mean']
                out[f'{angle_type}_std'] = stats['pred_std']
                out[f'{angle_type}_true'] = stats['true']
                out[f'{angle_type}_wraptrue'] = labels_dihedrals

                if f'flag_{angle_type}' in batch:
                    out.update({
                        f'{angle_type}_labels': stats['true'],
                        f'{angle_type}_flags': batch[f'flag_{angle_type}'].cpu().numpy()})
            return out

    @torch.no_grad()
    def get_classifier_preds(self, batch):
        p = self(batch)
        p = torch.sigmoid(p)
        out = {
            'pred_prob': p.cpu().numpy(),
            'label': batch['label'].cpu().numpy()
        }
        return out

    def load_partial_state_dict(self, partial_state_dict,
                                prefixes=None):
        if prefixes is None:
            prefixes = ['phi_gp', 'psi_gp', 'distance_gp',
                        'mlp_pep', 'mlp_hla',
                        'mlp_phi', 'mlp_psi', 'mlp_dist',
                        'layernorm_pep', 'layernorm_hla',
                        'nll_dihedrals', 'nll_dist']
        own_state = self.state_dict()
        for name, param in partial_state_dict.items():
            # if name not in own_state:
            # Skip parameters starting with the prefix
            if any([prefix in name for prefix in prefixes]):
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_partial_state_dict_from_checkpoint(self, path, device_name='cpu'):
        d = torch.load(path, map_location=torch.device(device_name))
        self.load_partial_state_dict(d['state_dict'])


if __name__ == '__main__':
    model = JointInterfaceModel(interface_pred=True)
    model.cuda()
    b = dict(
        peptide=torch.randn(7, 15+2).abs().round().long().cuda(),
        peptide_len=torch.tensor([8, 8, 8, 8, 8, 8, 8]).cuda(),
        hla_sequence=torch.randn(7, 34+2).abs().round().long().cuda())
    print(b)
