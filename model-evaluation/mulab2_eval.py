import sys
import caffe
import matplotlib
import numpy as np
import h5py
import argparse
import glob
import re
from collections import defaultdict

THRESHOLD = 0.5
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

# Example:
# python ../model-evaluation/mulab2_eval.py --model "snapshots/ONE-*_iter_10000.caffemodel" --proto "one-vs-all/deploy.prototxt" --hdf5 framesBGR_test_source.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model-glob', dest='model_pattern', type=str, required=True)
    parser.add_argument('--hdf5', type=str, required=True)
    args = parser.parse_args()

    true_positives = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_either = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_true = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_pred = np.zeros(NUMBER_OF_LABELS, dtype=int)
    num_correct = 0
    count = 0

    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    models = glob.glob(args.model_pattern)
    print models


    with open(args.hdf5,'r') as f:
        for hdf5File in f:
            hdf5File = hdf5File.rstrip()
            if not hdf5File: continue
            # read hdf5 and test all examples
            print "About to load: %s" % hdf5File
            h5file = h5py.File(hdf5File)
            data = h5file['data'][...]
            true_labels = h5file['label'][...].astype(bool)

            output = np.zeros((data.shape[0], len(models)))

            # data = data[1:100, :, :, :]
            # true_labels = true_labels[1:100]
            # output = np.zeros((data.shape[0], 37))


            for model in models:

                i = int(re.match(args.model_pattern.replace("*", "(\d+)"), model).group(1))

                net = caffe.Net(args.proto, model, caffe.TEST)
                caffe.set_mode_gpu()


                print "(%i) Starting to forward data..." % i
                out = net.forward_all(data=data)
                print "(%i) Prediction finished" % i
                prob_labels = out['prob']
                print prob_labels
                output[:, i] = prob_labels[:, 1]

            pred_labels = output > THRESHOLD


            print "SHAPES"
            print np.sum(pred_labels | true_labels, axis = 0).shape
            print num_either.shape

            print np.sum(pred_labels | true_labels, axis = 0)

            num_either = num_either + np.sum(pred_labels | true_labels, axis = 0)
            true_positives = true_positives + np.sum(pred_labels & true_labels, axis = 0)
            num_true += np.sum(true_labels, axis = 0)
            num_pred += np.sum(pred_labels, axis = 0)

            print "Comp:"
            print true_labels

            print sum(np.sum(np.logical_xor(pred_labels, true_labels), axis = 1) == 0)
            print sum(np.sum(true_labels, axis = 1) == 0)

            num_correct += sum(np.sum(np.logical_xor(pred_labels, true_labels), axis = 1) == 0)
            count += true_labels.shape[0]

            sys.stdout.write("\rAccuracy: %.1f%%" % (100.*sum(true_positives)/sum(num_either)))
            sys.stdout.flush()

        print true_positives
        print num_either

        print("\nResult: " + str(num_correct) + " out of " + str(count) + " were classified correctly")

        print ""
        print "Per label Precision / Recall:"

        for i in range(0, NUMBER_OF_LABELS):
            print "%d : %f \t %f" % (i, true_positives[i] * 100.0 / num_pred[i], true_positives[i] * 100.0 / num_true[i])

