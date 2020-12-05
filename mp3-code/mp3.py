# mp3.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import naive_bayes as nb
from sklearn.metrics import confusion_matrix

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels, dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    cm = confusion_matrix(dev_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    true_negative = tn
    false_positive = fp
    false_negative = fn
    true_positive = tp
    return accuracy, false_positive, false_negative, true_positive, true_negative


def main(args):
    #Modify stemming and lower case below. Note that our test cases may use both settings of the two parameters
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,stemming=False,lower_case=False)

    predicted_labels = nb.naiveBayes(train_set, train_labels, dev_set)
    accuracy, false_positive, false_negative, true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
    print("Accuracy:",accuracy)
    print("False Positive", false_positive)
    print("Fale Negative", false_negative)
    print("True Positive", true_positive)
    print("True Negative", true_negative)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Naive Bayes')
    parser.add_argument('--training', dest='training_dir', type=str, default = 'C:/Users/zhami/OneDrive/Desktop/cs440/mp3-code/template/MP3_data_zip/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = 'C:/Users/zhami/OneDrive/Desktop/cs440/mp3-code/template/MP3_data_zip/dev',
                        help='the directory of the development data')
    args = parser.parse_args()
    main(args)
