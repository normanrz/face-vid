from __future__ import generators
import sys
import caffe
import matplotlib
import numpy as np
import h5py
import glob
import re
import argparse
from collections import defaultdict
from frameIO import *
from frameset import *
from extract_frames import *

NUMBER_OF_LABELS = 37


def read_means(means_file_name):
    with io.open(means_file_name, "r") as f:
        s = f.read()
        return json.loads(s)


def avg_net_results(probs):
    return reduce(lambda x, y: x + y, probs) / 2


def get_predictions(video_file_name):

    means_file_name = "results/means"
    frames = get_frames(video_file_name)
    means = read_means(means_file_name)
    frameSets = preprocess_frames(frames, means)

    prob_result = avg_net_results([
        forward_net_single(frameSets[1], "nets/an-finetune/deploy.prototxt", "nets/ANF_iter_50000.caffemodel"),
        forward_net_multi(frameSets[1], "face-vid-nets/one-vs-all/deploy.prototxt", "face-vid-nets/snapshots/ONE-*_iter_10000.caffemodel")
    ])

    return prob_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--means', type=str, required=True)
    args = parser.parse_args()

    video_file_name = args.video
    means_file_name = args.means

    frames = get_frames(video_file_name)
    means = read_means(means_file_name)
    frameSets = preprocess_frames(frames, means)

    prob_result = avg_net_results([
        forward_net_single(frameSets[1].frames, "nets/an-finetune/deploy.prototxt", "nets/ANF_iter_50000.caffemodel"),
        forward_net_multi(frameSets[1].frames, "face-vid-nets/one-vs-all/deploy.prototxt", "face-vid-nets/snapshots/ONE-*_iter_10000.caffemodel")
    ])

    print prob_result
    label_result = prob_result > THRESHOLD
    print label_result


def preprocess_frames(frames, means):

    processId = id_generator()
    face_cache = {}
    labels = np.zeros((1,1,1,1))

    # Extract frames
    frameSet = FrameSet(frames, "framesOriginal", processId, labels)

    frameSets = split_grayscale_BGR(frameSet)
    frameSets = detect_faces_and_mask_surroundings(frameSets, face_cache)
    frameSets = induce_flows(frameSets)
    frameSets = filter_framesets_out_by_stream_name(frameSets, "grayscale")
    # frameSets = filter_frames_with_labels(frameSets)
    frameSets = resize_frames(frameSets, 227, 227)
    # frameSets = accumulate_means(frameSets, means, layer_counts)
    frameSets = transform_to_caffe_format(frameSets)

    # Finalize (e.g. mean substraction)
    frameSets = substract_means(frameSets, means)
    frameSets = set_mask_to_zero(frameSets)
    frameSets = normalize_frames(frameSets)
    frameSets = list(mark_as_test(frameSets, 0.9))
    frameSets = cross_flows(frameSets)

    return list(frameSets)

def forward_net_single(data, proto, model):
    net = caffe.Net(proto, model, caffe.TEST)
    # caffe.set_mode_gpu()

    print "Starting to forward data..."
    out = net.forward_all(data=data)
    print "Prediction finished"

    return out['prob']


def forward_net_multi(data, proto, model_pattern):

    models = glob.glob(model_pattern)

    # output = np.zeros((data.shape[0], len(models)))
    output = np.zeros((data.shape[0], NUMBER_OF_LABELS))

    for model in models:

        i = int(re.match(model_pattern.replace("*", "(\d+)"), model).group(1))

        net = caffe.Net(proto, model, caffe.TEST)
        # caffe.set_mode_gpu()

        print "(%i) Starting to forward data..." % i
        out = net.forward_all(data=data)
        print "(%i) Prediction finished" % i
        prob_labels = out["prob"]

        output[:, i] = prob_labels[:, 1]

    return output




if __name__ == "__main__":
    main()

