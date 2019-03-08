#!/usr/bin/env bash

OPENAI_LOGDIR=~/fixing_popart_logs/ddpg_old_$(date '+%d-%m-%Y_%H:%M:%S') \
mpiexec -n 6 python -m baselines.ddpg_old.main \
