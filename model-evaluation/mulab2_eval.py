import sys
import caffe
import matplotlib
import numpy as np
import h5py
import argparse
import glob
import re
from collections import defaultdict
from os import path

predict_path = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.append(predict_path)
from predict import forward_net_single, forward_net_multi

THRESHOLD = 0.2
NUMBER_OF_LABELS = 37


# Pretty print confusion table of predicted vs true labels
def print_table(headers, data):
    names = list(headers)
    row_format ="{:>6}" + "{:>5}" * len(names) + "{:>6}"
    print row_format.format("", *(names + ["[sum]"]))
    predictedSums = defaultdict(int)
    for l in names:
        row = []
        trueSum = 0
        for pl in names:
            if l == pl:
                row.append("(%d)" %matrix[(l,pl)])
            else:
                row.append(matrix[(l,pl)])
            trueSum += matrix[(l,pl)]
            predictedSums[pl] += matrix[(l,pl)]
        row.append(trueSum)
        print row_format.format(l, *row)

    row = []
    for pl in names:
        row.append(predictedSums[pl])
    print row_format.format("[sum]", *(row + [sum(row)]))


def avg_model_results(m1, m2):
    return (m1 + m2) / 2


def mulab(hdf5list, model_callback):
    true_positives = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_either = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_true = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_pred = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_correct = 0
    count = 0

    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()


    with open(hdf5list, 'r') as f:
        for hdf5File in f:
            hdf5File = hdf5File.rstrip()
            if not hdf5File: continue
            # read hdf5 and test all examples
            print "About to load: %s" % hdf5File
            h5file = h5py.File(hdf5File)
            data = h5file['data'][...]
            true_labels = h5file['label'][...].astype(bool)

            # data = data[1:100, :, :, :]
            # true_labels = true_labels[1:100]

            output = model_callback(data)

            pred_labels = output > THRESHOLD


            # print "SHAPES"
            # print np.sum(pred_labels | true_labels, axis = 0).shape
            # print num_either.shape

            # print np.sum(pred_labels | true_labels, axis = 0)

            num_either = num_either + np.sum(pred_labels | true_labels, axis = 0)
            true_positives = true_positives + np.sum(pred_labels & true_labels, axis = 0)
            num_true += np.sum(true_labels, axis = 0)
            num_pred += np.sum(pred_labels, axis = 0)

            # print "Comp:"
            # print true_labels

            # print sum(np.sum(np.logical_xor(pred_labels, true_labels), axis = 1) == 0)
            # print sum(np.sum(true_labels, axis = 1) == 0)

            num_correct += sum(np.sum(np.logical_xor(pred_labels, true_labels), axis = 1) == 0)
            count += true_labels.shape[0]

            sys.stdout.write("\rAccuracy: %.1f%%" % (100.*sum(true_positives)/sum(num_either)))
            sys.stdout.flush()

        print("\n")
        print true_positives
        print num_either

        print("\nResult: " + str(num_correct) + " out of " + str(count) + " were classified correctly")

        print ""
        print "Per label Precision / Recall:"

        for i in range(0, NUMBER_OF_LABELS):
            precision = 0 if num_pred[i] == 0 else true_positives[i] * 100.0 / num_pred[i]
            recall = 0 if num_true[i] == 0 else true_positives[i] * 100.0 / num_true[i]
            print "%d : %f \t %f" % (i, precision, recall)

        print ""
        print "Overall Precision / Recall / Accuracy"

        overall_precision = 0 if np.sum(num_pred) == 0 else np.sum(true_positives) * 100.0 / np.sum(num_pred)
        overall_recall = 0 if np.sum(num_true) == 0 else np.sum(true_positives) * 100.0 / np.sum(num_true)
        overall_accuracy = np.sum(true_positives) * 100.0 / np.sum(num_either)
        print "%f \t %f \t %f" % (overall_precision, overall_recall, overall_accuracy)


if __name__ == "__main__":

    hdf5file = "face-vid-nets/flows_test_source.txt"

    def model_callback(data):
        # return avg_model_results(
        #     forward_net_single(data, "nets/an-finetune/deploy.prototxt", "nets/ANF_iter_50000.caffemodel"),
        #     forward_net_multi(data, "face-vid-nets/one-vs-all/deploy.prototxt", "face-vid-nets/snapshots/ONE-*_iter_10000.caffemodel")
        # )
        return forward_net_single(data, "face-vid-nets/flow/deploy.prototxt", "face-vid-nets/snapshots/FLOW-_iter_20000.caffemodel")
        # return forward_net_single(data, "face-vid-nets/an-finetune/deploy.prototxt", "face-vid-nets/snapshots/ANF-S_iter_35000.caffemodel")
        return forward_net_multi(data, "face-vid-nets/one-vs-all/deploy.prototxt", "face-vid-nets/snapshots/ONE-*_iter_2500.caffemodel")

    mulab(hdf5file, model_callback)
