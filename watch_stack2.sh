#!/usr/bin/env bash

SEED=42

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

date_string=$(date '+%m-%d-%Y_%H:%M:%S')
part1_dir=/tmp/logs/stack_2_SPARSE_demo_horizon_50_SEED_${SEED}_${date_string}

OPENAI_LOGDIR=${part1_dir} \
mpiexec -n 1 python -m ddpg_curiosity_mc_her.ddpg.main \
--env-id 'FetchStack2SparseStage3-v1' \
--do-evaluation 'True' \
--render-eval 'False' \
--render-training 'False' \
--boxpush-heatmaps 'True' \
--map-dynamics-loss 'True' \
--seed ${SEED} \
--train-policy-fn 'noisy_exploit' \
--eval-policy-fn 'greedy_exploit' \
--agent-roles 'exploit' \
--memory-type 'replay_buffer' \
--exploit-layers 3 \
--exploit-hidden 256 \
--explore-layers 3 \
--explore-hidden 256 \
--dynamics-layers 3 \
--dynamics-hidden 256 \
--exploit-Q-lr '0.001' \
--exploit-pi-lr '0.001' \
--exploit-critic-l2-reg '0' \
--explore-Q-lr '0.001' \
--explore-pi-lr '0.001' \
--explore-critic-l2-reg '1e-2' \
--dynamics-lr '0.007' \
--exploit-polyak-tau '0.001' \
--explore-polyak-tau '0.05' \
--exploit-use-layer-norm 'False' \
--explore-use-layer-norm 'True' \
--exploit-gamma 'auto' \
--explore-gamma 'auto' \
--episode-time-horizon '50' \
--buffer-size '1e6' \
--n-epochs 2000 \
--n-cycles 50 \
--n-batches 40 \
--batch-size 1024 \
--rollout-batches-per-cycle 8 \
--rollout-batch-size 1 \
--n-test-rollouts 50 \
--exploit-noise-type 'normal_0.04' \
--explore-noise-type 'adaptive-param_0.1, normal_0.04' \
--exploit-normalize-returns 'True' \
--exploit-popart 'True' \
--explore-normalize-returns 'True' \
--explore-popart 'True' \
--agents-normalize-observations 'True' \
--agents-normalize-goals 'True' \
--dynamics-normalize-observations 'True' \
--use-her 'True' \
--replay-strategy 'future' \
--replay-k 4 \
--mix-extrinsic-intrinsic-objectives-for-explore '0.5,0.5' \
--sub-goal-divisions '[[0,1,2],[3,4,5]]' \
--do-demo-only "True" \
--restore-from-ckpt "trained_models/stack2/model.ckpt" \

