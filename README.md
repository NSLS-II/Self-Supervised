# A Self-supervised workflow for particle picking in Cryo-EM
#Authors: Donal McSweeney donal.mcs15@gmail.com 
#qun.liu@gmail.com
#Referecne:  https://doi.org/10.1107/S2052252520007241
# 

## System requirement
1. Anaconda2 with cuda/GPU/tensorflow/keras support
2. Relion2 or Relion3 compiled with GPU support

## Quickstart
Download the Self-Supervised to the directory under relion main directory

## Prepare ctf corrected micrograph star file, rename it as below
CtfFind/micrographs_defocus_ctf.star

### Pre-processing
1) Configure settings in `config.ini`
2) Run `python build_preprocess.py`. This will generate a bash script for initial (local) picking and 2D class averaging.
3) Run `sh preprocess.sh`

### Self-Supervised training/particle picking
1) Configure settings in `config.ini`
2) Run <code> python build_workflow.py</code>. This will generate a bash file for iterative training/picking.
3) Run <code> sh workflow.sh</code>

### Picking all micrigraphs: Create empty directory of Kpicker/aligned, then run
python Self-Supervised/kpicking_cpu.py --input_dir 'aligned' --output_dir 'Kpicker/aligned' --coordinate_suffix '_kpicker' --threshold 0.9  --threads 10 --particle_size 260  --bin_size 4

### using Localpicker for initial particle picking based on shapes
for mrcfile in `(ls */*.mrc)`; do 
python -W ignore Self-Supervised/localpicker.py  --mrc_file=${mrcfile} --particle_size=240 --bin_size=9  --threshold=0.0015 --max_sigma=10
done 

