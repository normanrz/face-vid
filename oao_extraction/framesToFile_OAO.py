from __future__ import generators
from natsort import natsorted
import os, sys, re

FILENAME_RE = re.compile(r"^frame-bgr")
LABEL_RE = re.compile(r"_([0-9]+[A-Z]*)")

def get_all_filenames(root_dir,):

  # read the content of the root directory and filter all directories
  directory_names = map(lambda f: os.path.join(root_dir, f), os.listdir(root_dir))
  directories = filter(os.path.isdir, directory_names)

  filenames = []

  for directory in directories:
    for parent_dir, sub_dirs, files in os.walk(directory):

      # sort files
      for filename in natsorted(files):
        if (filename.endswith(("jpeg", "jpg", "png"))):
          if re.search(FILENAME_RE, filename):
            absolute_file = os.path.join(parent_dir, filename)
            filenames.append(absolute_file)
  return filenames


def get_all_labels(filenames):

  labels = dict()

  # First step: gather all labels
  for filename in filenames:
    for label in extract_label_from_filename(filename):
      labels[label] = 0

  # Assign integer label to FACS unit
  i = 0
  for label in natsorted(labels):
    labels[label] = i
    i += 1

  return labels


def extract_label_from_filename(filename):
  return re.findall(LABEL_RE, filename)


def map_labels(filenames, labels):

  print ("Number of labels: %s" % len(labels))

  for filename in filenames:
    first_label = extract_label_from_filename(filename)[0]
    yield (filename, labels[first_label])


def write_pairs_to_disk(root_dir, filename_label_pairs):

  # open ouput file
  filename = os.path.join(root_dir, "filelist.txt")
  output_file = open(filename, "w")

  for filename, label in filename_label_pairs:
    line = '{} {}\n'.format(filename, label)
    output_file.write(line)

  # close output file
  output_file.close()


def write_labels_to_disk(root_dir, labels):

  # open ouput file
  filename = os.path.join(root_dir, "labelmapping.txt")
  output_file = open(filename, "w")

  output_file.write("Mapping FACS label -> integer label\n")
  for key in sorted(labels, key=labels.get):
    line = '{} {}\n'.format(key, labels[key])
    output_file.write(line)

  # close output file
  output_file.close()


if __name__ == "__main__":

  if len(sys.argv) < 1:
    sys.exit("Usage: %s <frames_directory>" % sys.argv[0])

  root_dir = os.path.abspath(sys.argv[1])

  if (not os.path.isdir(root_dir)):
    sys.exit("The argument <root directory> is not a valid directory.")

  filnames = get_all_filenames(root_dir)
  labels = get_all_labels(filnames)
  filename_label_pairs = map_labels(filnames, labels)

  write_pairs_to_disk(root_dir, filename_label_pairs)
  write_labels_to_disk(root_dir, labels)

