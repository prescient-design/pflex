#!/usr/bin/env bash

set -eux

git clone https://github.com/snerligit/mhc-pep-threader.git
wget -c http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64 -O clustalo
mv clustalo ./mhc-pep-threader/
chmod +x ./mhc-pep-threader/clustalo
chmod +x ./mhc-pep-threader/main.py
export PYTHONPATH=$PYTHONPATH:$(pwd)/mhc-pep-threader
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/gen_ensemble

# recommend installing pyrosetta through the wheel file from here https://www.pyrosetta.org/downloads 

if command -v mamba
then
 mamba env create --file env.yaml
else
 conda env create --file env.yaml
fi

conda activate pmhc_flex
pip install -e .

