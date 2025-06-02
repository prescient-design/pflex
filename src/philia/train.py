import os
import random
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from philia.trainval_data.loading_utils import get_group_index, get_indices_from_file, load_pdb_split, load_pdb_split_bulk, get_df_indices, add_relax
from philia import trainval_data
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import logging
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('highest')


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    # make log dir
    # pathlib.Path(cfg.wandb.dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(**cfg.wandb, save_dir=cfg.best_model_path)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config['cwd'] = os.path.join(cfg.best_model_path)
    wandb.init(
        config=wandb_config,
        dir=cfg.best_model_path,
        **cfg.wandb,)
    model = instantiate(cfg.model)
    # wandb.watch(model, log='all')
    dataset = instantiate(cfg.data)
    dataset.configure_alphabet(
        alphabet_hla=model.alphabet_hla,
        alphabet_peptide=model.alphabet_peptide)
    if not cfg.model.interface_pred:
        train_idx, valid_idx, test_idx = get_group_index(
            indices=np.arange(len(dataset)),
            group_ratio=[0.95, 0.025, 0.025],
            seed=cfg.seed
        )
        # import pdb; pdb.set_trace()
        train_set = Subset(dataset, list(train_idx))
        valid_set = Subset(dataset, list(valid_idx))
        weights = (dataset.df).loc[train_idx, 'label'].values*4 + 1
        train_sampler = WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True)
        train_loader = DataLoader(
            train_set, batch_size=cfg.optim.batch_size, sampler=train_sampler,
            pin_memory=True,
            num_workers=4)
        valid_loader = DataLoader(
            valid_set, batch_size=cfg.optim.val_batch_size, shuffle=False
        )
    else:
        # train_idx = get_indices_from_file(
        #     '/home/parj2/stage/tcr_design/train_test_splits/train.txt', dataset.df, index_type='pdbid')
        test_split_path = os.path.join(trainval_data.__path__[0], 'structural_splits/case2/testset.txt')
        # valid_split_path = os.path.join(trainval_data.__path__[0], cfg.val_split)
        test_pdbids = load_pdb_split(test_split_path)
        # print(cfg.val_split)
        if cfg.val_pdb_only:
            valid_pdbids = load_pdb_split_bulk(cfg.val_split, os.path.join(trainval_data.__path__[0], 'structural_splits/case2'), val_pdb_only=True)
        else:
            valid_pdbids = load_pdb_split_bulk(cfg.val_split, os.path.join(trainval_data.__path__[0], 'structural_splits/case2'))
            # print('before adding relaxed structures:{}'.format(len(valid_pdbids)))
            # valid_pdbids = add_relax(valid_pdbids, num=3)
            # print('after adding relaxed structures:{}'.format(len(valid_pdbids)))
        train_pdbids = list(set(list(dataset.df['pdbid'].values)) - set(test_pdbids) - set(valid_pdbids))
        test_idx = get_df_indices(dataset.df, test_pdbids)
        valid_idx = get_df_indices(dataset.df, valid_pdbids)
        train_idx = get_df_indices(dataset.df, train_pdbids)
        logger.info(f"num_train: {len(train_idx)}, num_val: {len(valid_idx)}, num_test: {len(test_idx)}")
        logger.info(f"pdbid_train: {len(train_pdbids)}, pdbid_val: {len(valid_pdbids)}, pdbid_test: {len(test_pdbids)}")
        train_set = Subset(dataset, list(train_idx))
        valid_set = Subset(dataset, list(valid_idx))
        train_loader = DataLoader(train_set, batch_size=cfg.optim.batch_size, shuffle=True) #, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=cfg.optim.val_batch_size, shuffle=False)
        if cfg.checkpoint_path is not None:
            model.load_partial_state_dict_from_checkpoint(cfg.checkpoint_path)
    split = {
        'valid_idx': valid_idx.tolist(),
        'test_idx': test_idx.tolist()}
    wandb.config.update(split)
    if str(cfg.val_split)[0] != '[':
        cv_split = cfg.val_split.split('/')[-1].split('.')[0]
    else:
        cv_split = '_'.join(['_'.join(x.split('/')[-2:]).split('.')[0] for x in cfg.val_split])
    best_model_path = train_model(
        model,
        train_loader,
        valid_loader,
        wandb_logger,
        # hyperparameter tuning
        os.path.join(cfg.best_model_path, os.path.join(f"val_split-{cv_split}", f"bs-{cfg.optim.batch_size}-lr-{cfg.model.lr}-dmlp-{cfg.model.d_mlp}-w_dihderal-{cfg.model.weight_dihedral_loss}-w_aug-{cfg.model.weight_aug}-w_length-{cfg.model.weight_length}-scale-{cfg.model.scale_option}")),
        trainer_kwargs=cfg.trainer
    )
    logger.info(f'best model path: {best_model_path}')
    wandb.finish()


def train_model(model,
                train_loader,
                valid_loader,
                logger,
                store_dir,
                trainer_kwargs={}):
    """
    batch_size: size
    """
    # n_train_batches = len(train_loader)
    early_stop_callback = EarlyStopping(monitor="val_loss_early_stopping",
                                        # min_delta=min_delta,
                                        patience=200,
                                        verbose=False,
                                        mode="min")
    checkpoint_callback = ModelCheckpoint(
        mode="min",
        dirpath=store_dir,
        filename='checkpoint-{epoch}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,)
    learning_rate_callback = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[early_stop_callback,
                   checkpoint_callback,
                   learning_rate_callback],
        # val_check_interval=n_train_batches//2,
        **trainer_kwargs)
    trainer.fit(model, train_loader, valid_loader)
    return trainer.checkpoint_callback.best_model_path


if __name__ == '__main__':
    main()
