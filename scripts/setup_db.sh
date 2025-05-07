#!/usr/bin/env bash

# the following needs to be exported if not already done so
#export PYTHONPATH=$PYTHONPATH: <fill the mhc-pep-threader>
#export PYTHONPATH=$PYTHONPATH: <fill the src/gen_ensemble>

conda activate pmhc_flex

python ../database/scripts/fetch_pmhcs_from_pdb.py