#!/usr/bin/env bash

#OPENAI_LOGDIR=~/boxpush_logs/intrinsic_maze$(date '+%d-%m-%Y_%H:%M:%S') python -m baselines.her.experiment.train \
#--num_cpu 6 \
#--env 'boxpush-v0' \
#random_eps=0.8 \
#noise_eps=0.1 \
#intrinsic_motivation_eps=1.0 \
#--n_epochs=30 \
#--replay_strategy 'future' \
#use_layer_norm=False \
#use_param_noise=False \
#param_noise_stddev=0.02 \
#rollout_batches_per_cycle=1 \
#batch_size=256 \
#buffer_size=1000000

OPENAI_LOGDIR=~/boxpush_logs/intrinsic_maze$(date '+%d-%m-%Y_%H:%M:%S') python -m baselines.her.experiment.train \
--num_cpu 6 \
--env 'boxpush-v0' \
random_eps=0.0 \
noise_eps=0.2 \
intrinsic_motivation_eps=0.5 \
--n_epochs=50 \
--replay_strategy 'future' \
use_layer_norm=True \
use_param_noise=True \
param_noise_stddev=0.02 \
rollout_batches_per_cycle=1 \
batch_size=256 \
buffer_size=1000000
