import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from collections import defaultdict

from gen_ensemble.utils.file_ops import *
from gen_ensemble.utils.template import *
from gen_ensemble.flexpepdock.fpd import *
from gen_ensemble.constants.constants import *
from gen_ensemble.analyses.run_analyses import *
from gen_ensemble.homology_model.homology_model import *
from gen_ensemble.cluster_pep_bbs.cluster_pipeline import *
from gen_ensemble.analyses.compare_with_md_ensembles_with_natives import *

# Load the Rosetta commands for use in the Python shell
from pyrosetta import *

@hydra.main(config_path="../hydra_config", config_name="pipeline")
def run(cfg:DictConfig):

    orchestrate_cfg = hydra.utils.instantiate(cfg.orchestrator)
    homology_model_cfg = hydra.utils.instantiate(cfg.homology_model)
    fpd_cfg = hydra.utils.instantiate(cfg.flexpepdock)

    o = Orchestrator(orchestrate_cfg, homology_model_cfg, fpd_cfg)

    if orchestrate_cfg.md.compare_with_native:
        o.run_md_comparison_analyses()

    if orchestrate_cfg.benchmarks.run_benchmark and orchestrate_cfg.run.pred_cst:
        init(options=f"-cst_fa_weight {fpd_cfg.cst_fa_weight} -cst_weight {fpd_cfg.cst_weight}")
        o.orchestrate_inf_runner_for_benchmark_set()

    if orchestrate_cfg.run.unk and orchestrate_cfg.run.pred_cst:
        init(options=f"-cst_fa_weight {fpd_cfg.cst_fa_weight} -cst_weight {fpd_cfg.cst_weight}")
        o.orchestrate_inf_runner_for_unk_sequences()

class Orchestrator:

    def __init__(self, orchestrate_cfg:DictConfig, homology_model_cfg:DictConfig, fpd_cfg:DictConfig):

        self.orchestrate_cfg = orchestrate_cfg
        self.homology_model_cfg = homology_model_cfg
        self.fpd_cfg = fpd_cfg
        self.pdbs_to_fold = []
        self.vars = defaultdict(list)

    def load_testset_ids(self):

        # read the test set pdbids
        lines = ''
        with open(self.orchestrate_cfg.run.testset, 'r') as testinfilehndlr:
            lines += testinfilehndlr.readline().rstrip()

        self.pdbs_to_fold = lines.split(',')

    def load_unk_sequences(self):

        # read the test set sequences
        df = pd.read_csv(self.orchestrate_cfg.run.unk_sequences)

        for idx, row in df.iterrows():
            self.pdbs_to_fold.append(row['peptide']+"_"+row['mhc'])

    def load_and_read_inferred_csts(self):

        # load the inferred csts
        ds = np.load(self.orchestrate_cfg.run.inferred_csts, allow_pickle=True)

        # read the inferred distances
        self.vars['distances_mean'] = (ds.item().get('distances_mean')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PSEUDO_MHC_RES_LENGTH))
        self.vars['distances_std'] = (ds.item().get('distances_std')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PSEUDO_MHC_RES_LENGTH))

        # DIHEDRAL DISTANCE CONSTRAINTS

        if self.orchestrate_cfg.initialize.dihedral_ctype == "CIRCULAR_HARMONIC":
            # read the inferred phi dihedrals
            self.vars['phi_mean'] = (ds.item().get('phi_mean')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, 1))
            self.vars['phi_std'] = (ds.item().get('phi_std')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, 1))

            # read the inferred psi dihedrals
            self.vars['psi_mean'] = (ds.item().get('psi_mean')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, 1))
            self.vars['psi_std'] = (ds.item().get('psi_std')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, 1))

        elif self.orchestrate_cfg.initialize.dihedral_ctype == "CIRCULAR_SPLINE":
            # read the inferred phi dihedrals
            self.vars['phi_mean'] = (ds.item().get('phi_mean')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PROB_VALUES_FOR_SPLINE))
            self.vars['phi_std'] = (ds.item().get('phi_std')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PROB_VALUES_FOR_SPLINE))

            # read the inferred psi dihedrals
            self.vars['psi_mean'] = (ds.item().get('psi_mean')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PROB_VALUES_FOR_SPLINE))
            self.vars['psi_std'] = (ds.item().get('psi_std')).reshape((len(self.pdbs_to_fold), PEP_LENGTH, PROB_VALUES_FOR_SPLINE))

    def orchestrate_inf_runner_for_benchmark_set(self):

        self.load_testset_ids()
        self.load_and_read_inferred_csts()

        self.orchestrate_cfg.output.update({"outdir": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir})

        if not os.path.exists(self.orchestrate_cfg.output.outdir):
            os.system(f"mkdir {self.orchestrate_cfg.output.outdir}")

        fasta = FASTA(self.orchestrate_cfg.fasta)

        for i in range(0, len(self.pdbs_to_fold)):
            
            pdbid = self.pdbs_to_fold[i]

            target = os.path.join(self.orchestrate_cfg.dir, pdbid+"_reordered.pdb")
            allele = fasta.alleles[pdbid]
            peptide = fasta.pep_seq[pdbid]
            template = []
            template_files = []
            mhc_chains = []
            pep_chains = []

            print ("For pdbid: ", pdbid)
            print ("Previous template: ", template)

            if self.orchestrate_cfg.initialize.multi_template:

                template_files, mhc_chains, pep_chains, template = read_templates_from_multi_template_lib(self.orchestrate_cfg.multi_template_dir, len(peptide), fasta)

            elif self.orchestrate_cfg.initialize.cst_based_template:

                self_template = pdbid
                template = find_nearest_template_based_on_dist_dih(i, self.vars, self_template, self.orchestrate_cfg.dir, fasta)
                print ("New template: ", template)
            else:
                template_for_each_pdb = closest_template_for_each_pdb_based_on_sequence(self.orchestrate_cfg.dir, fasta.pep_chains, fasta.pep_seq, self.pdbs_to_fold)
                template = template_for_each_pdb[pdbid]
            
            if len(template) > 0 or len(template_files) > 0:

                if len(template) > 0:

                    mhc_chains = [fasta.mhc_chains[template[0]]]
                    pep_chains = [fasta.pep_chains[template[0]]]
                    template_files = [os.path.join(self.orchestrate_cfg.dir, template+"_reordered.pdb")]
                    new_template = [item + "_reordered.pdb" for item in template]
                    template = new_template


                for t in range(0, len(template_files)):
                    self.homology_model_cfg.update({"template_file": template_files[t]})
                    self.homology_model_cfg.update({"template_pdb": template[t]})
                    self.homology_model_cfg.update({"mhc_chain": mhc_chains[t]})
                    self.homology_model_cfg.update({"peptide_chain": pep_chains[t]})
                    self.homology_model_cfg.update({"outdir": os.path.join(self.orchestrate_cfg.output.outdir, pdbid, template[t].replace(".pdb", ""))})
                    self.homology_model_cfg.update({"allele": allele})
                    self.homology_model_cfg.update({"peptide": peptide})

                    starting_model = homology_model_target_on_template(self.homology_model_cfg)

                    self.fpd_cfg.update({"input_model": starting_model})
                    if not (starting_model == ''):
                        fpd = FPD(self.fpd_cfg, self.vars, i)
                        fpd.distance_constraints(peptide)
                        fpd.dihedral_constraints(peptide, self.orchestrate_cfg.initialize.dihedral_ctype)
                        filetag = fpd.fpd_apply()

                        compare_docked_with_native(self.homology_model_cfg.outdir, self.orchestrate_cfg.dir, filetag)
                        compare_MD_and_docked_ensembles(self.orchestrate_cfg.output.outdir)

    def orchestrate_inf_runner_for_unk_sequences(self):

        self.load_unk_sequences()
        self.load_and_read_inferred_csts()

        self.orchestrate_cfg.output.update({"outdir": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir})

        if not os.path.exists(self.orchestrate_cfg.output.outdir):
            os.system(f"mkdir {self.orchestrate_cfg.output.outdir}")

        fasta = FASTA(self.orchestrate_cfg.fasta)
        template_for_each_pdb = closest_template_for_each_pdb_based_on_sequence(self.orchestrate_cfg.dir, fasta.pep_chains, fasta.pep_seq, self.pdbs_to_fold)

        for i in range(0, len(self.pdbs_to_fold)):
            
            pid = self.pdbs_to_fold[i]
            allele = pid.split("_")[1]
            peptide = pid.split("_")[0]
            template = template_for_each_pdb[pid]

            if self.orchestrate_cfg.initialize.cst_based_template:

                template = find_nearest_template_based_on_dist_dih_unk_seqs(i, self.vars, peptide, self.orchestrate_cfg.dir, fasta)
                print ("Selected template: ", template)
            
            if template:

                mhc_chain = fasta.mhc_chains[template]
                pep_chain = fasta.pep_chains[template]

                template_file = os.path.join(self.orchestrate_cfg.dir, template+"_reordered.pdb")

                self.homology_model_cfg.update({"template_file": template_file})
                self.homology_model_cfg.update({"template_pdb": template+"_reordered.pdb"})
                self.homology_model_cfg.update({"mhc_chain": mhc_chain})
                self.homology_model_cfg.update({"peptide_chain": pep_chain})
                self.homology_model_cfg.update({"outdir": os.path.join(self.orchestrate_cfg.output.outdir, pid)})
                self.homology_model_cfg.update({"allele": allele})
                self.homology_model_cfg.update({"peptide": peptide})

                starting_model = homology_model_target_on_template(self.homology_model_cfg)

                self.fpd_cfg.update({"input_model": starting_model})
                if not (starting_model == ''):
                    fpd = FPD(self.fpd_cfg, self.vars, i)
                    fpd.distance_constraints(peptide)
                    fpd.dihedral_constraints(peptide, self.orchestrate_cfg.initialize.dihedral_ctype)
                    filetag = fpd.fpd_apply()

                    run_cluster_pipeline(self.homology_model_cfg.outdir, self.fpd_cfg.peptide_chain, self.orchestrate_cfg)

    def run_md_comparison_analyses(self):

        fasta = FASTA(self.orchestrate_cfg.fasta)
        select_conserved_mhc_structures_from_MD_ensemble_and_measure_rms(self.orchestrate_cfg, fasta.mhc_chains, fasta.pep_chains)

if __name__ == "__main__":

    # Load Rosetta database files
    run()
