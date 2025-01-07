import os
import sys
import pandas as pd
import csv

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    result = f"all params: {all_param} || trainable params: {trainable_params} || trainable%: {100 * trainable_params / all_param}"
    return result


def get_file_count(directory):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    return file_count


def sort_directories_by_file_count(base_path):
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    directories_file_counts = [(d, get_file_count(os.path.join(base_path, d))) for d in directories]
    directories_file_counts.sort(key=lambda x: x[1], reverse=True)
    return directories_file_counts


import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

def get_apcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest bpcer at which the probability of apcer == op
    @rtype: float
    """
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]

def get_bpcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest apcer at which the probability of bpcer == op
    @rtype: float
    """
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]

def performances_compute(prediction_scores, gt_labels, threshold_type='eer', op_val=0.1, verbose=True):
    # fpr = apcer, 1-tpr = bpcer
    # op_val: 0 - 1
    # gt_labels: list of ints,  0 for attack, 1 for bonafide
    # prediction_scores: list of floats, higher value should be bonafide
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1)
    bpcer = 1 - tpr
    val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
    val_auc = auc(fpr, tpr)

    if threshold_type=='eer':
        threshold = eer_threshold
    elif threshold_type=='apcer':
        _, _, threshold = get_apcer_op(fpr, bpcer, threshold, op_val)
    elif threshold_type=='bpcer':
        _, _, threshold = get_bpcer_op(fpr, bpcer, threshold, op_val)
    else:
        threshold = 0.5

    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    threshold_APCER = type2 / num_fake
    threshold_BPCER = type1 / num_real
    threshold_ACER = (threshold_APCER + threshold_BPCER) / 2.0

    if verbose is True:
        print(f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')

    return val_auc, val_eer, [threshold, threshold_APCER, threshold_BPCER, threshold_ACER]

def evalute_threshold_based(prediction_scores, gt_labels, threshold):
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER

def evaluate_mad_performance(prediction_scores, gt_labels, verbose=False):
    # Calculate ROC curve metrics
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1)
    bpcer = 1 - tpr
    apcer = fpr

    # Get EER
    eer, _, _ = get_eer_threhold(fpr, tpr, threshold)

    # Calculate APCER at fixed BPCER points
    _, apcer_bpcer20, _ = get_bpcer_op(apcer, bpcer, threshold, 0.20)
    _, apcer_bpcer10, _ = get_bpcer_op(apcer, bpcer, threshold, 0.10)
    _, apcer_bpcer1, _ = get_bpcer_op(apcer, bpcer, threshold, 0.01)

    # Calculate BPCER at fixed APCER points
    _, bpcer_apcer20, _ = get_apcer_op(apcer, bpcer, threshold, 0.20)
    _, bpcer_apcer10, _ = get_apcer_op(apcer, bpcer, threshold, 0.10)
    _, bpcer_apcer1, _ = get_apcer_op(apcer, bpcer, threshold, 0.01)

    # Calculate AUC
    auc_score = auc(fpr, tpr)

    results = {
        "auc_score": auc_score,
        "eer": eer,
        "apcer_bpcer20": apcer_bpcer20,
        "apcer_bpcer10": apcer_bpcer10,
        "apcer_bpcer1": apcer_bpcer1,
        "bpcer_apcer20": bpcer_apcer20,
        "bpcer_apcer10": bpcer_apcer10,
        "bpcer_apcer1": bpcer_apcer1,
    }

    if verbose:
        print("\nMAD Performance Metrics:")
        print(f"AUC: {results['auc_score']:.4f}")
        print(f"EER: {results['eer']:.4f}")
        print("\nAPCER at fixed BPCER:")
        print(f"APCER@BPCER20%: {results['APCER@BPCER20%']:.4f}")
        print(f"APCER@BPCER10%: {results['apcer_bpcer10']:.4f}")
        print(f"APCER@BPCER1%: {results['apcer_bpcer1']:.4f}")
        print("\nBPCER at fixed APCER:")
        print(f"BPCER@APCER20%: {results['bpcer_apcer20']:.4f}")
        print(f"BPCER@APCER10%: {results['bpcer_apcer10']:.4f}")
        print(f"BPCER@APCER1%: {results['bpcer_apcer1']:.4f}")

    return results


def write_scores(img_paths, prediction_scores, gt_labels, output_path):
    """
    Write detection scores to CSV file
    Args:
        img_paths: List of image paths
        prediction_scores: List of prediction scores
        gt_labels: List of ground truth labels
        output_path: Path to save the CSV file
    """
    try:
        # Verify all inputs have same length
        lengths = [len(img_paths), len(prediction_scores), len(gt_labels)]
        if len(set(lengths)) != 1:
            print(f"Warning: Length mismatch - Paths: {lengths[0]}, Predictions: {lengths[1]}, Labels: {lengths[2]}")
            length = min(lengths)
        else:
            length = lengths[0]

        # Prepare data for writing
        save_data = []
        for idx in range(length):
            save_data.append({
                'image_path': img_paths[idx],
                'label': str(gt_labels[idx]).replace(' ', ''),
                'prediction_score': prediction_scores[idx]
            })

        # Write to CSV
        with open(output_path, mode='w', newline='') as csv_file:
            fieldnames = ['image_path', 'label', 'prediction_score']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in save_data:
                writer.writerow(data)

        print(f'Successfully saved prediction scores in {output_path}')

    except Exception as e:
        print(f"Error in write_scores: {str(e)}")
        raise
