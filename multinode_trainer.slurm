#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---


#SBATCH --job-name=torchtitan_multi_node

#SBATCH --ntasks=2

#SBATCH --nodes=2

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=104

#SBATCH --partition=gpu

#SBATCH --output=logs/slurm-%j.out

#SBATCH --wait-all-nodes=1

#SBATCH --nodelist=g356,g364

echo "SLURM script has started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:/home/ubuntu/git/torchtitan

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
# export FI_PROVIDER="efa"
# Ensure that P2P is available
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
# export NCCL_NET_GDR_LEVEL=0

# debugging flags (optional)
export PYTHONFAULTHANDLER=1
# optional debug settings
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# export NCCL_NET_GDR_LEVEL=2

# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-12.6
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
# export CUDA_LAUNCH_BLOCKING=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/libnvvp:/usr/local/lib

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# export NCCL_SOCKET_IFNAME="ib0@ibp26s0"
# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond,bond0"
# export NCCL_SOCKET_IFNAME=enp157s0f0np0
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
# export NCCL_IB_HCA=mlx5

# export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1

export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1

export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0

# export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
# export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}

# dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./torchtitan/train.py --job.config_file ${CONFIG_FILE}
# dcgmi profile --resume
