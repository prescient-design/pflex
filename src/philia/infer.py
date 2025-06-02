import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl
from philia.trainval_data.loading_utils import get_group_index, get_indices_from_file
from philia.analysis.export import combine_dict
from philia.trainval_data.loading_utils import get_group_index, get_indices_from_file, load_pdb_split, get_df_indices, add_relax, load_pdb_split_bulk
from philia import trainval_data
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig
import logging
from time import time
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # os.makedirs('/scratch/site/u/parj2/pmhc_pretrain', exist_ok=True)
    start = time()
    os.makedirs(os.path.dirname(os.path.realpath(cfg.infer.out_path)), exist_ok=True)
    dataset = instantiate(cfg.data)
    model = instantiate(cfg.model)
    # own_state = model.state_dict()
    # print("Model")
    # for name, param in own_state.items():
    #     print(name)
    dataset.configure_alphabet(
        alphabet_hla=model.alphabet_hla,
        alphabet_peptide=model.alphabet_peptide)
    if not cfg.model.interface_pred:
        _, _, test_idx = get_group_index(
            indices=np.arange(len(dataset)),
            group_ratio=[0.95, 0.025, 0.025],
            seed=cfg.seed
        )
        test_set = Subset(dataset, list(test_idx))
        # test_set = Subset(dataset, list(test_idx))
        test_loader = DataLoader(
            test_set, batch_size=cfg.infer.batch_size, shuffle=False
        )
    else:
        # valid_pdbids = load_pdb_split_bulk(cfg.val_split, os.path.join(trainval_data.__path__[0], 'splits'))
        # valid_pdbids = add_relax(valid_pdbids, num=3)
        test_split_path = os.path.join(trainval_data.__path__[0], cfg.infer.test_path) #'structual_splits/case2/testset.txt')#'splits/testset.txt')
        test_pdbids = load_pdb_split(test_split_path)
        test_idx = get_df_indices(dataset.df, test_pdbids)
        logger.info(f"num_test_pdbids: {len(test_pdbids)}, num_test_idx: {len(test_idx)}")
        if cfg.infer.test_indices is not None:
            test_idx = get_indices_from_file(
                cfg.infer.test_indices, dataset.df, cfg.infer.index_type)
        dataset = Subset(dataset, list(test_idx))
        test_loader = DataLoader(dataset, batch_size=cfg.infer.batch_size, shuffle=False)
        # model.load_partial_state_dict_from_checkpoint(cfg.infer.checkpoint_path,
        #                                               cfg.infer.device_name)
    # print("Checkpoint")
    # for name, param in checkpoint['state_dict'].items():
    #     print(name)
    # checkpoint = torch.load()

    print("Loading full state dict")
    checkpoint = torch.load(
        cfg.infer.checkpoint_path,
        map_location=torch.device(cfg.infer.device_name))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # split = {
    #     'valid_idx': valid_idx.tolist(),
    #     'test_idx': test_idx.tolist()}
    trainer = pl.Trainer(accelerator=cfg.infer.device_name)
    predictions = trainer.predict(model, test_loader)
    full_dict = combine_dict(predictions)
    np.save(cfg.infer.out_path, full_dict, allow_pickle=True)
    print('runtime: {}'.format(time()-start))

if __name__ == '__main__':
    main()
