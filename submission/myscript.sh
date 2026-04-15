#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn
module load cuda

./gpu-life 5120 5000 /scratch/$USER/output.5120.5000.gpu
./gpu-life 5120 5000 /scratch/$USER/output.5120.5000.gpu
./gpu-life 5120 5000 /scratch/$USER/output.5120.5000.gpu
