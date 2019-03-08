# from tensorflow.python.summary import event_accumulator as ea
# from tensorflow.python.summary import event_multiplexer as em

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
plt.style.use('classic')

import numpy as np
import pandas as pd

# event_acc = EventAccumulator('/Users/JB/Desktop/test_fetch_push_logs/HYBRID_0p5_0p5_epsilon_greedy_noisy_explore_23-10-2018_04:46:37/tb')

event_acc = EventMultiplexer().AddRunsFromDirectory('/Users/JB/Desktop/test_fetch_push_logs/')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Runs().keys())
# print(event_acc.Tags())

# # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('test/success_rate'))


import seaborn as sns
sns.set(style="darkgrid")

# Load an example dataset with long-form data