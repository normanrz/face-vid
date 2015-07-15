import numpy as np
import h5py
import sys

NUMBER_OF_LABELS = 37

# python count_labels.py framesBGR_test_source.txt
if __name__ == "__main__":
  np.set_printoptions(threshold=np.nan)

  hdf5list = sys.argv[1]
  output = np.zeros(NUMBER_OF_LABELS, dtype="int")

  with open(hdf5list, 'r') as f:
    for hdf5File in f:
      hdf5File = hdf5File.rstrip()
      if not hdf5File: continue
      # read hdf5 and test all examples
      print "About to load: %s" % hdf5File
      h5file = h5py.File(hdf5File)
      labels = h5file['label'][...].astype(bool)

      output = output + np.sum(labels, axis=0)

  print output