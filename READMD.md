# P-flex

This work can be used to model ensembles of pMHC-I molecules with peptides of lengths 8-13 amino acids. 

Note: The following instructions are for generating structural ensembles given inferences from the ML component of P-flex

## Installation

1. Navigate to the **pmhc-flex** folder after dowloading from gitlab.
2. From within **pmhc-flex** folder, run install script 
```
chmod +x ./scripts/install.sh
./scripts/install.sh
```
The install script will clone **mhc-pep-threader**, download **clustalo** and set up appropriate python paths. Please double check to make sure the paths are setup correctly.
3. Download and install pyrosetta through the wheel file from here https://www.pyrosetta.org/downloads.
4.  If the installation proceeds without issues, then all the necessary libaries should be installed to run your first simulation.
5. Download the **database** folder from zenodo and place it under **pmhc-flex** folder.

## Generating ensembles for examples (unknown sequences and benchmarks)

If you want to run benchmarks, please download the **md_analyses** folder from zenodo and place it inside **pmhc-flex** dir. In the **example** folder, we have the following dirs:

### short 

```unk.csv``` - file containing peptide and MHC sequences for which we want to generate ensembles
```test.npy``` - file containing inferences for pMHC sequences in ***unk.csv***
```test.txt``` - file containing PDBIDs to run benchmarks on

### long

```unk.csv``` - file containing peptide and MHC sequences for which we want to generate ensembles
```test.npy``` - file containing inferences for pMHC sequences in ***unk.csv***
```test.txt``` - file containing PDBIDs to run benchmarks on

### outputs

The folders inside outputs were runs we performed for example cases:

1. short_unk_cst - generate ensembles for short length peptides in ***short/unk.csv*** where the examples are treated as an unknown pMHC pair. Constraint-based templates are used as starting models.
2. short_unk_seq - generate ensembles for short length peptides in ***short/unk.csv*** where the examples are treated as an unknown pMHC pair. Sequence-based templates are used as starting models.
3. short_benchmark_cst - generate ensembles for short length peptides in ***short/test.txt*** where the examples are treated as an unknown pMHC pair. Constraint-based templates are used as starting models. 
4. short_benchmark_seq  - generate ensembles for short length peptides in ***short/test.txt*** where the examples are treated as an unknown pMHC pair. Sequence-based templates are used as starting models. 
5. long_unk - generate ensembles for short length peptides in ***long/unk.csv*** where the examples are treated as an unknown pMHC pair. Multiple templates are used as starting models.

Explanation of output files:

1. Inside ***LLFGYPVYV_A*02:01*** folder, you will find subfolder (6PTE_reordered) highlighting the template used. For instance, for this peptide, 6PTE was used as a starting model.
2. Within ***6PTE_reordered***, 
    a. Inputs to the ***mhc-pep-threader***, mhc_list and pep_list.
    b. run command to run ***mhc-pep-threader***.
    c. Outputs produced by ***mhc-pep-threader***, ```6PTE_reordered*.pdb``` are cleaned and trimmed templates for homology modeling, clustal input and output files, Rosetta alignment (grishin) and restricted relax (movemap) files, homology model, relaxed homology model and binding energy of the relaxed model.
    d. FPD models with substring "dock_refined" in the pdb file name.
3. Inside benchmark runs, we have additional analyses files:
    a. <pbdid>_scores.csv: a csv file containing rosetta scores and comparison with native scores
    b. md_model_comparison.tsv: a tsv file containing n (FPD models) x 300 (md samples) comparison scores.

## Parameters

This codebase uses hydra. So the parameters are pretty easy to update. All the config files are in ```pmhc_flex/src/gen_ensemble/hydra_config```. You can update the parameters according to your desired values based on the use case. Below are the explanations of the parameters in each of the hydra yaml file:

1. orchestrator/default.yaml

```

# initialize parameters

initialize: 
  distance_ctype - # function to adjust the energy landscape based on distance constraints; default is FLAT_HARMONIC
  dihedral_ctype - # function to adjust the energy landscape based on dihedral constraints; default is CIRCULAR_HARMONIC
  cst_based_template - Selected templates based on predicted constraints

benchmarks:
  run_benchmark - set this flag to true if you want to run benchmarks

run:
  unk - set this flag to true if the pMHC you want to model is not a benchmark (or an unknown sequence for which you want to generate models)
  pred_cst - set this flag to true if you want to generate ensembles using constraints (in this version of the code, we do not support running or generating ensembles without using any constraints)

  unk_sequences - path to sequence list. See examples to learn the formats
  testset - path to PDBID list for benchmarks. See examples to learn the formats
  inferred_csts - path to inferred csts from ML component. See examples to learn the formats (the npy files are output by the ML component, so we do not need to edit this file)

output:
  outdir - path to output dir, will be automatically updated inside the code.

fasta - path to PDBS fasta file containing peptide and MHC sequences in the database/lib
dir- path to PDB files in database/lib/TRIMMED_TEMPLATES
multi_template_dir - path multiple templates in database/lib/multi_templates for longer length peptides

md:
  compare_with_native - compate MD frames with native structures; default to false since these are precomputed and make available in the md_analyses folder
  dir - path MD frames
  rms_to_native - path to file containing comparison metrics between MD frames and natives

```

2. homology_model/default.yaml

```
path_to_bin - path to mhc-pep-threader
nstruct - number of structures to relax; default to 1
mhcs - list of MHC alleles to model for each peptide; see examples for formats
peptides - list of peptides to model; see examples for formats
pep_start_index - start index of the peptide; default to 181 
interface_cutpoint - end index of the MHC; default to 180
relax_after_threading - refine models after threading; default to true

template_file - template file used for homology modeling; will be automatically updated by the orchestrator
template_pdb - template pdbid used for homology modeling; will be automatically updated by the orchestrator
mhc_chain - mhc chain ID of the template pdb; will be automatically updated by the orchestrator
peptide_chain - peptide chain ID of the template pdb; will be automatically updated by the orchestrator
outdir - path to the output directory where homology models are placed; will be automatically updated by the orchestrator
allele - input allele; will be automatically updated by the orchestrator
peptide - input peptide; will be automatically updated by the orchestrator
clustal_path - path to the clustal omega binary

```

3. flexpepdock/default.yaml

```
receptor_chain - receptor (HLA) chain; default to chain A
peptide_chain - peptide chain; default to chain B
pep_refine - run fpd refine; default to true
nstruct - number of models to output; default to 3 for the purpose of testing, please increase this number when sampling more conformers
suffix - suffix for models; default to _dock_refined
cst_fa_weight - full-atom constraint weights, default to 0.1
cst_weight - constraint weights, default to 0.1

input_model - input model; will be automatically updated by the orchestrator

```

## Running examples:

From within the examples folder, run the following command:

```
python ../src/gen_ensemble/orchestrator/orchestrate.py
```

## Caveats
1. In the paper, the benchmark was done by directly tuning FPD temperature in the Rosetta source code due to no availability of parameters interfacing the user to tune the temperature. If there is sufficient interest in this, then please contact us and we can assist in updating the source to reflect the temperature change from 0.2 to 0.8.
2. If you want to cluster fdp and md ensembles, you can invoke cluster_pipeline by placing the FDP and MD models together in a dir and invoking the method with that dir as input.

## Notes

You can also update the database by running ```setup_db.sh``` script from within **pmhc_flex/scripts**