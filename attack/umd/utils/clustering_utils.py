from __future__ import absolute_import
from __future__ import print_function

import torch
import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np


def compute_score(P, A, mode='mean'):
    # Get the weights for core nodes
    core = P[0]
    core_size = len(core)
    if core_size < 2:
        sys.exit('Core node less than 2!')
    A_core = A[core, :]
    A_core = A_core[:, core]
    # print(core)
    if mode == 'mean':
        score_core = np.sum(A_core) / (core_size * (core_size - 1))
    elif mode == 'min':
        score_core = np.min(np.sum(A_core, axis=1) / (core_size - 1))
    else:
        sys.exit('Wrong mode!')

    # Get the max connection for periphery nodes
    A_preph = A[core, :]
    A_preph = np.delete(A_preph, core, axis=1)
    score_preph = np.max(np.mean(A_preph, axis=0))

    return score_core - score_preph, score_core, score_preph


def compute_score_combined(P, A, A_single, mode='mean'):
    # Get the weights for core nodes
    core = P[0]
    core_size = len(core)
    if core_size < 2:
        sys.exit('Core node less than 2!')
    A_core = A[core, :]
    A_core = A_core[:, core]
    # print(core)
    if mode == 'mean':
        score_core = np.sum(A_core) / (core_size * (core_size - 1))
    elif mode == 'min':
        score_core = np.min(np.sum(A_core, axis=1) / (core_size - 1))
    else:
        sys.exit('Wrong mode!')

    # Get the max connection for periphery nodes
    A_preph = A_single[core, :]
    A_preph = np.delete(A_preph, core, axis=1)
    score_preph = np.max(np.mean(A_preph, axis=0))

    return score_core - score_preph, score_core, score_preph

