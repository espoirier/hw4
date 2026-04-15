#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn
module load cuda

./gpu-life 100 100 /scratch/$USER/output.100.100.gpu
./life 100 100 /scratch/$USER/output.100.100.cpu
