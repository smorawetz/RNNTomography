import sys
import numpy as np

fname = sys.argv[1]

data = np.loadtxt(fname)

data[:, 1] = np.square(data[:, 1])

np.savetxt(fname, data)
