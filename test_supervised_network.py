import sys
sys.path.insert(0, "/home/pezzotto/lib/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-i686.egg")

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import math

import crnn
import pyublas

data_in = np.loadtxt("/home/pezzotto/tmp/input.txt", dtype=np.float64)
data_out = np.loadtxt("/home/pezzotto/tmp/output.txt", dtype=np.float64)
data_out = np.array(data_out,  ndmin=2).T

pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "r")
net = cPickle.load(pickle_file)
print "Got a net with ",net.input_size," inputs, ",net.hidden_size," hiddens, ", net.output_size, " outputs"
pickle_file.close()

nhiddens = net.hidden_size

net.x = np.random.rand(net.size,1)
errs = []
netout = []
for x,y in zip(data_in, data_out):
    
    outs = net(np.array(x))    
    netout.append(outs)
    errs.append(math.fabs( netout[-1] - y))

plt.plot(netout, label="Network")
plt.plot(data_out, label="Data")
plt.title("Avg. Error: " + str(sum(errs)/len(errs)))
plt.legend(loc="best")

plt.figure()
plt.plot(errs)
plt.title("Avg. Error: " + str(sum(errs)/len(errs)))


plt.show()
