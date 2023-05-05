import os
import sys
import time

import numpy as np
import argparse

##############################
#        PARAMETERS          #
##############################

# RESULT_DIR = "./nc_mnist/" + args.path  # directory for storing results
IMG_FILENAME_TEMPLATE = '%s_%s_label_%d.npy'  # image filename template for visualization results

NUM_CLASSES = 10  # total number of classes in the model

##############################
#      END PARAMETERS        #
##############################


def outlier_detection(l1_norm_list, idx_mapping):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    pass

def analyze_pattern_norm_dist(result_dir, template, dataset, num_classes):
    mask_flatten = []
    idx_mapping = {}
    for y_label in range(num_classes):
        mask_filename = template % (dataset, 'mask', y_label)
        if os.path.isfile('%s/%s' % (result_dir, mask_filename)):
            mask = np.load('%s/%s' % (result_dir, mask_filename))

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))

    outlier_detection(l1_norm_list, idx_mapping)

    pass

def outlier_detection_plabel(l1_norm_list, idx_mapping, plabel):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    flag_list = []

    for y_label in idx_mapping:
        if y_label == plabel:
            print("'plabel index: ", np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad)
    pass

def analyze_pattern_norm_dist_plabel(result_dir, template, dataset, num_classes, plabel):
    mask_flatten = []
    idx_mapping = {}
    for y_label in range(num_classes):
        mask_filename = template % (dataset, 'mask', y_label)
        if os.path.isfile('%s/%s' % (result_dir, mask_filename)):
            mask = np.load('%s/%s' % (result_dir, mask_filename))

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]
    outlier_detection_plabel(l1_norm_list, idx_mapping, plabel)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./path",
        help="",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        help="",
    )
    args = parser.parse_args()

    RESULT_DIR = "./path/" + args.path  # directory for storing results

    start_time = time.time()
    analyze_pattern_norm_dist()
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)