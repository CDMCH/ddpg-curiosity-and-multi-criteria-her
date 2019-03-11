#!/usr/bin/env bash

SEED=3

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

GPU_MESSAGE="Set environment variable ALL_GPUS to train with GPUs.\n"
GPU_EXAMPLE="For example, if you have two GPUs with device numbers 0 and 1, to use both of them, set ALL_GPUS to \"[0, 1]\".\nIf you have one GPU, set ALL_GPUS to \"[0]\".\n"

printf "ALL_GPUS is set to \"${ALL_GPUS}\"\n"

if [ -z ${ALL_GPUS+x} ]; then printf "${GPU_MESSAGE}"; printf "${GPU_EXAMPLE}"; ALL_GPUS='none'; fi

date_string=$(date '+%m-%d-%Y_%H:%M:%S')
part1_dir=logs/stack_2_SPARSE_full_curriculum_\(pt1_trainer_easy\)_curiosity_multi_criteria_horizon_50_SEED_${SEED}_${date_string}
part2_dir=logs/stack_2_SPARSE_full_curriculum_\(pt2_stacking_curriculum\)_curiosity_multi_criteria_horizon_50_SEED_${SEED}_${date_string}
part3_dir=logs/stack_2_SPARSE_full_curriculum_\(pt3_test\)_curiosity_multi_criteria_horizon_50_SEED_${SEED}_${date_string}

OPENAI_LOGDIR=${part1_dir} \
mpiexec -n 8 python -m ddpg_curiosity_mc_her.ddpg.main \
--env-id 'FetchStack2SparseTrainerEasy-v1' \
--do-evaluation 'True' \
--render-eval 'False' \
--render-training 'False' \
--boxpush-heatmaps 'True' \
--map-dynamics-loss 'True' \
--seed ${SEED} \
--train-policy-fn 'half_noisy_exploit_half_noisy_explore' \
--eval-policy-fn 'greedy_exploit' \
--agent-roles 'exploit, explore' \
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
--exploit-noise-type 'adaptive-param_0.1, normal_0.04' \
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
--stop-at-score 0.85 \
--save-checkpoints-at '[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]' \
--split-gpu-usage-among-device-nums "$ALL_GPUS"


OPENAI_LOGDIR=${part2_dir} \
mpiexec -n 8 python -m ddpg_curiosity_mc_her.ddpg.main \
--env-id 'FetchStack2Sparse-v1' \
--do-evaluation 'True' \
--render-eval 'False' \
--render-training 'False' \
--boxpush-heatmaps 'True' \
--map-dynamics-loss 'True' \
--seed ${SEED} \
--train-policy-fn 'half_noisy_exploit_half_noisy_explore' \
--eval-policy-fn 'greedy_exploit' \
--agent-roles 'exploit, explore' \
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
--exploit-noise-type 'adaptive-param_0.1, normal_0.04' \
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
--restore-from-ckpt ${part1_dir}/saved_model/model.ckpt \
--save-at-score 0.90 \
--save-checkpoints-at '[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]' \
--stop-at-score 0.95 \
--split-gpu-usage-among-device-nums "$ALL_GPUS"

OPENAI_LOGDIR=${part3_dir} \
mpiexec -n 8 python -m ddpg_curiosity_mc_her.ddpg.main \
--env-id 'FetchStack2SparseTest-v1' \
--do-evaluation 'True' \
--render-eval 'False' \
--render-training 'False' \
--boxpush-heatmaps 'True' \
--map-dynamics-loss 'True' \
--seed ${SEED} \
--train-policy-fn 'half_noisy_exploit_half_noisy_explore' \
--eval-policy-fn 'greedy_exploit' \
--agent-roles 'exploit, explore' \
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
--exploit-noise-type 'adaptive-param_0.1, normal_0.04' \
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
--restore-from-ckpt ${part2_dir}/saved_model/model.ckpt \
--save-at-score 0.80 \
--save-checkpoints-at '[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]' \
--split-gpu-usage-among-device-nums "$ALL_GPUS"
