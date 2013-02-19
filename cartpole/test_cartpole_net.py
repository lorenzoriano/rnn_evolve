import cPickle
import sys

sys.path.insert(0, "..")

import crnn
import pyublas
import crnn.libcartpole
import numpy as np
import matplotlib.pyplot as plt

import crnn.libcartpole
test_net = crnn.libcartpole.test_net

cp = crnn.libcartpole

pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "r")
net = cPickle.load(pickle_file)
pickle_file.close()

#net.x = np.random.rand(*net.x.shape)
print net
cartpole = cp.SingleCartPole(True)
cartpole.FORCE_MAG = 10.0
cartpole.reset(False)

#stabilising
net.randomiseState()
for i in xrange(10):
    net(cartpole.getObservation())

all_obs = []
all_forces = []
step = 0
while not cartpole.outsideBounds():
    obs = cartpole.getObservation()
    all_obs.append(obs)
    
    out = net(obs)[0]

    all_forces.append(out)
    cartpole.doAction(out)
    step += 1
    
    if step > 600:
        break

#print "C Code fitness: ",  test_net(net, 10)


print "Steps: ",  step
all_obs = np.array(all_obs)

plt.figure()
plt.subplot(2, 2, 3)
plt.plot(all_obs[:, 0])
plt.title("CartPos")

plt.subplot(2, 2, 1)
plt.plot(all_obs[:, 2])
plt.title("PoleAngle 1")

plt.subplot(2, 2, 2)
plt.plot(all_obs[:, 4])
plt.title("PoleAngle 2")

plt.subplot(2, 2, 4)
plt.plot(all_forces)
plt.title("Forces")

plt.show()
