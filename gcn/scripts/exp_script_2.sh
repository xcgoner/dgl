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


basename=/homes/cx2/gcn/dgl-gcn/gcn/results/exp_script_2

watchfile=$basename.log


modelfile=$basename.params


DGLBACKEND=mxnet python /homes/cx2/gcn/dgl-gcn/gcn/gcn_edge_origin.py --save $modelfile --dataset "cora" --lr 0.005 --n-epochs 200 --n-layers 2 --n-hidden 64 --dropout 0.8 --normalization 'sym' --self-loop 2>&1 | tee $watchfile
