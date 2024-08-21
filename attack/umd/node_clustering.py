from __future__ import absolute_import
from __future__ import print_function

import torch
import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.clustering_utils import compute_score, compute_score_combined
from scipy import special
import json

parser = argparse.ArgumentParser(description='Test transferability of estimated perturbation')
parser.add_argument("--mode", default="patch", type=str)
parser.add_argument("--RUN", default=1, type=int)
parser.add_argument("--ATTACK", default="patch", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
parser.add_argument("--DATASET", default="mnist", type=str)
parser.add_argument("--shards", default=1, type=int)
parser.add_argument("--slices", default=1, type=int)
parser.add_argument("--sub", default=0, type=int)
parser.add_argument("--checkpoint_root", default="./path", type=str)
args = parser.parse_args()

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)

config["SUB_MODEL"] = args.sub
config["RUN"] = args.RUN
if args.DEVICE != -1:
    config["DEVICE"] = "cuda:" + str(args.DEVICE)
else:
    config["DEVICE"] = "cpu"
config["SETTING"] = "A2O"
if args.ATTACK == "patch":
    config["PATTERN_TYPE"] = args.ATTACK
if args.DATASET == "cifar10" or args.DATASET == "mnist" or args.DATASET == "fmnist" or args.DATASET == "imagenet":
    config["DATASET"] = args.DATASET

if config["DATASET"] == "cifar10" or config["DATASET"] == "mnist" or config["DATASET"] == "fmnist":
    config["NUM_CLASS"] = 10
elif config["DATASET"] == "imagenet":
    config["NUM_CLASS"] = 100

NC = config["NUM_CLASS"]  
N_init = 5
def get_threshold(N, conf=0.95):
    return np.sqrt(2) * special.erfinv(2 * np.power(conf, (1 / N)) - 1)
modelpath = os.path.join(args.checkpoint_root, str(config["RUN"]), "SNO_" + str(args.shards), "cache", "shard-" + str(config["SUB_MODEL"]) + '-0.pt')
ckpt_path = './color_maps/{}/{}_{}/{}/{}'.format(config['DATASET'], args.shards, args.slices, config["RUN"], config['SUB_MODEL'])
print("Detect: {}, Dataset: {}, Mode: {}, Type: {},  Run: {}".format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
RED_path = './estimated/{}/{}_{}/{}/{}'.format(config['DATASET'], args.shards, args.slices, config["RUN"], config['SUB_MODEL'])

path = os.path.join(args.checkpoint_root, str(config["RUN"]))
setting = np.load(os.path.join(path, "setting.npy"), allow_pickle=True)
backdoor_indexes = setting[:-1]
config['TC'] = setting[-1]

if args.ATTACK != "clean":
    source = np.arange(stop=config["NUM_CLASS"])
    source = np.delete(source, config['TC'])
    poisoned_pairs = [[i, config['TC']]for i in source]

print("Expected pairs: ")
print(poisoned_pairs)
scores = []
pairs_set = []
theta_set = []
# Get adjacency matrix
stat_evals = []
H_score = []
pert_null = []
pert_eval = []
pairs = []
trans_graph = []
for t in range(NC):
    for s in range(NC):
        pairs.append([s, t])
        if s != t:
            trans = torch.load(os.path.join(ckpt_path, 'color_map_{}_{}'.format(s, t)))
            trans = np.transpose(trans)     # The color map is originally stored with each role the same source class
        else:
            trans = np.zeros((NC, NC))
        trans = np.reshape(trans, (NC*NC))  # (0, 0), (0, 1), ..., (0, 9), (1, 0), (1, 1), ...
        trans_graph.append(trans)
pairs = np.asarray(pairs)
trans_graph = np.asarray(trans_graph)
trans_graph_single = trans_graph
trans_graph_mutual = (trans_graph + np.transpose(trans_graph)) / 2

# Remove class pairs (0, 0), (1, 1), ...
idx = 0
idx_remove = []
for i in range(len(pairs)):
    if pairs[i][0] == pairs[i][1]:
        idx_remove.append(idx)
    idx += 1
# Reshape graph
trans_graph_single = np.delete(trans_graph_single, idx_remove, axis=0)
trans_graph_single = np.delete(trans_graph_single, idx_remove, axis=1)
trans_graph_mutual = np.delete(trans_graph_mutual, idx_remove, axis=0)
trans_graph_mutual = np.delete(trans_graph_mutual, idx_remove, axis=1)
pairs = np.delete(pairs, idx_remove, axis=0)      # remove labels of class pairs (0, 0), (1, 1), ...

# Vertices, edges
V_names = pairs
A_single = trans_graph_single
A_mutual = trans_graph_mutual
V = np.arange(start=0, stop=len(V_names), dtype=int)


# Get initial community
A_flatten = A_mutual.flatten()
rank = np.flip(np.argsort(A_flatten))
init_candidate = []
for i in range(len(rank)):
    pair1 = int(rank[i] / len(V))
    pair2 = int(rank[i] % len(V))
    pair1_name = V_names[pair1]
    pair2_name = V_names[pair2]
    if pair1_name[0] != pair2_name[0]: # Source class should be different
        init_candidate.append([pair1, pair2])
    if len(init_candidate) == N_init:
        break

core_best_global = None
score_best_global = - float("inf")
for init in init_candidate:
    # Initialize the core-periphery structure
    core_record = [np.array(init)]
    P = [np.array(init), np.delete(V, np.array(init))]
    score, score_mul, score_sin = compute_score_combined(P, A_mutual, A_single, mode='min')
    score_record = [score]
    score_mul_record = [score_mul]
    score_sin_record = [score_sin]
    converge = False
    while not converge:
        core_best = None
        score_best = - float("inf")
        # Trial include every node in the periphery to the core
        core_old = core_record[-1]
        preph_old = np.delete(V, core_old)
        for i in range(len(preph_old)):
            # Skip if the source class already exists
            s = V_names[preph_old[i]][0]
            skip = False
            for j in range(len(core_old)):
                if s == V_names[core_old[j]][0]:
                    skip = True
                    break
            if skip:
                continue
            core_trial = np.concatenate([core_old, [preph_old[i]]])
            preph_trial = np.delete(preph_old, i)
            P_trial = [core_trial, preph_trial]
            score_trial, score_mul, score_sin = compute_score_combined(P_trial, A_mutual, A_single, mode='min')
            if score_trial > score_best:
                score_best = score_trial
                core_best = core_trial
                score_mul_best = score_mul
                score_sin_best = score_sin
        if core_best is None:
            converge = True
        else:
            core_record.append(core_best)
            score_record.append(score_best)
            score_mul_record.append(score_mul_best)
            score_sin_record.append(score_sin_best)
    if np.max(score_record) > score_best_global:
        score_best_global = np.max(score_record)
        core_best_global = core_record[np.argmax(score_record)]
        core_last = core_record[-1]

core = core_best_global
preph = np.delete(V, core)
P = [core, preph]
score = score_best_global
H_score.append(score)
pairs_detected = pairs[core]
print("Detected pairs: ")
print((pairs_detected.tolist()))
np.save(os.path.join(ckpt_path, 'pairs_detected.npy'), pairs_detected)
# Plot sorted adjacency matrix
order = []
for i in range(len(P)):
    for j in range(len(P[i])):
        order.append(P[i][j])
A_single = A_single[order, :]
A_single = A_single[:, order]
plt.imshow(A_single, cmap='hot', vmin=0, vmax=1)
plt.colorbar()
plt.axis('off')
plt.savefig(os.path.join(ckpt_path, 'color_map_all.png'))
plt.close()
pairs_detected = np.load(os.path.join(ckpt_path, 'pairs_detected.npy'))
pairs_idx = pairs_detected[:, 0] * NC + pairs_detected[:, 1]
pairs_set.append(pairs_idx)

score = []
# Get pert/patch norm
pert_size_eval = []
pert_size_null = []
for t in range(NC):
    for s in range(NC):
        if s != t:
            if args.mode == "patch":
                patch = torch.load(os.path.join(RED_path, 'mask_{}_{}'.format(s, t)))
                patch_size = torch.sum(torch.abs(patch)).item()
                if np.where(pairs_idx == s * NC + t)[0] > 0:
                    pert_size_eval.append(patch_size)
                else:
                    pert_size_null.append(patch_size)
            elif args.mode == "pert":
                pert = torch.load(os.path.join(RED_path, 'pert_{}_{}'.format(s, t)))
                pert_size = torch.norm(pert).item()
                if np.where(pairs_idx == s * NC + t)[0] > 0:
                    pert_size_eval.append(pert_size)
                else:
                    pert_size_null.append(pert_size)
stat_eval = np.asarray(pert_size_eval)
stat_null = np.asarray(pert_size_null)

pert_eval.append(pert_size_eval)
pert_null.append(pert_size_null)

stat_eval = 1 / stat_eval
stat_eval = np.median(stat_eval)
stat_evals.append(stat_eval)
stat_null = 1 / stat_null

med = np.median(stat_null)
MAD = np.median(np.abs(stat_null - med))
scores.append((stat_eval - med) / (MAD * 1.4826))
theta = get_threshold(len(stat_null), conf=0.95)
theta_set.append(theta)

print("Null size statistic")
print([np.median(pert_size_null) for pert_size_null in pert_null])
print("Eval size statistic")
print([np.median(pert_size_eval) for pert_size_eval in pert_eval])
print("Pert reverse size stat is")
print(stat_evals)
print("Threshold is: ")
print(theta_set)
print("Score is: ")
print(scores)
num_detected = 0
ind = []
for i in range(len(scores)):
    if scores[i] > theta_set[i]:
        num_detected += 1
print("Number of detected models: {}".format(num_detected))

stat_evals = np.asarray(stat_evals)
H_score = np.asarray(H_score)
theta_set = np.asarray(theta_set)
scores = np.asarray(scores)
pairs_set = np.asarray(pairs_set)
if args.ATTACK != "clean":
    stat = scores
    pairs_idx = pairs_set[np.argmax(stat)]
    poisoned_pairs = np.array(poisoned_pairs)
    poisoned_pairs_idx = poisoned_pairs[:, 0] * NC + poisoned_pairs[:, 1]
    num = 0
    for i in pairs_idx:
        if i in poisoned_pairs_idx:
            num += 1
    print("# of detected pairs: {}".format(num))




