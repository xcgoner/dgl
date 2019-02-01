#!/bin/bash
#PBS -l select=5:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_dgl


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=68


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;


basename=/homes/cx2/gcn/dgl-gcn/gcn/results/exp_script_1

watchfile=$basename.log


# logfile=/homes/cx2/federated/results/exp_unbalanced_lr_15_reg_000.txt

# > $logfile


DGLBACKEND=mxnet python /homes/cx2/gcn/dgl-gcn/gcn/gcn_concat.py --dataset "cora" --n-epochs 200 --n-layers 10 --normalization 'sym' --self-loop 2>&1 | tee $watchfile
