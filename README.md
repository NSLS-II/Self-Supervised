# A self-supervised workflow for particle picking in cryo-EM
#Authors: Donal McSweeney donal.mcs15@gmail.com 
#qun.liu@gmail.com
#Reference:  https://doi.org/10.1107/S2052252520007241
# 

## System requirement
1. Anaconda2 with cuda/GPU/tensorflow/keras support
2. Relion2 or Relion3 compiled with GPU and MPI support.

## Quickstart
Download the Self-Supervised to a directory under the relion project directory structure as it uses relion for
2D class averaging. 

## Give an alias name for the directory containing motion corrected micrographs. 
`ln -s /path/to/motition-corrected/micrographs  aligned`

## Do ctf correction, select ctf corrected micrographs for iterative training and picking, rename its star file as below
`CtfFind/micrographs_defocus_ctf.star`

### Pre-processing. picked particles will be under local/aligned/
1) Configure settings in `Self-Supervised/config.ini`
2) Run `python Self-Supervised/build_preprocess.py`. This will generate a bash script for initial (local) picking and 2D class averaging.
3) Run `sh preprocess.sh`

### Self-Supervised training/particle picking. picked particles will be under kpicker/aligned/
1) Configure settings in `Self-Supervised/config.ini`
2) Run `python Self-Supervised/build_workflow.py`. This will generate a bash file for iterative training/picking.
3) Run `sh workflow.sh`

### Picking all micrographs: Create an empty directory of Kpicker/aligned if it does not exist, then run
python Self-Supervised/kpicking_cpu.py --input_dir 'aligned' --output_dir 'Kpicker/aligned' --coordinate_suffix '_kpicker' --threshold 0.9  --threads 10 --particle_size 260  --bin_size 4

### using Localpicker for initial particle picking based on shapes
for mrcfile in `(ls */*.mrc)`; do ;
python -W ignore Self-Supervised/localpicker.py  --mrc_file=${mrcfile} --particle_size=260 --bin_size=9  --threshold=0.0015 --max_sigma=10;
done

