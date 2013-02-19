import cPickle
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "/home/pezzotto/lib/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-i686.egg")

import crnn
import pyublas
import crnn.libcartpole
import numpy as np
import matplotlib.pyplot as plt

test_net = crnn.libcartpole.test_net_single_learning

cp = crnn.libcartpole
markov = False

pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "r")
net = cPickle.load(pickle_file)
pickle_file.close()

print "Got a network with ",  net.hidden_size,  " hidden nodes"

#net.learning_factor = np.array([0.48271407286880935, 0.36937227856642374, 0, 0.39657426746276381, 0.160501860602068, 0.25312406110481533, 0.32500264682681168, 0.54672926678649381, 0.11984763474297443, 0.89361661964169248, 0.077670618719603346])
#net.signs = 2.*np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0] ) -1 

first_W = net.W[net.input_size:, :].ravel().copy()


try:
    print "LEARNING: ",  net.learning_factor
    print "SIGNS: ",  net.signs
except AttributeError:
    print " ...no"
    pass

class PID:
    def __init__(self):
        self.previous_error = 0
        self.integral = 0
    def __call__(self,  error,  dt):
        kp = 3.0
        kd = 0.1
        ki = 1.0

        self.integral += (error*dt)
        derivative = (error - self.previous_error)/dt
        output = (kp*error) + (ki*self.integral) + (kd*derivative)
        self.previous_error = error
        return output

#net.learning = False

if __name__ == "__main__":

    cartpole = cp.SingleCartPole(markov)
    cartpole.TAU = 0.02
    cartpole.FORCE_MAG = 100.0
    print "Cartpole TAU is ",  cartpole.TAU
    print "Cartpole FORCE_MAG is ",  cartpole.FORCE_MAG
    
#    cartpole.failureAngle = 0.78539816339744828 #45 degres

    net.randomiseState()
    step = 0
    avg_steps = 0   
    
    for sgh in xrange(50):
        cartpole.reset(True)
        net.randomiseState()
        for ablsh in xrange(10):
            net(cartpole.getObservation())
        
        while not cartpole.outsideBounds():
            obs = cartpole.getObservation()            
            out = net(obs)[0]
            cartpole.doAction(out)
            step += 1        
            if step > 5000:
                break
         
        print "trial: ",  sgh,  " steps: ",  step,  "Avg diff_w: ",  np.average(np.fabs(first_W - net.W[net.input_size:, :].ravel()))
        avg_steps += step
        step = 0

    avg_steps = float(avg_steps) / 50.0

    all_obs = []
    all_forces = []

    step = 0

    cartpole.reset(True)
    net.randomiseState()
    for ablsh in xrange(10):
        net(cartpole.getObservation())

    pid = PID()
    while not cartpole.outsideBounds():
        obs = cartpole.getObservation()
        all_obs.append(obs)
        
        net_out = net(obs)[0]        
        alpha = 0.8
        cart,  pole = cartpole.getObservation()  - 0.5
        pid_input = alpha*pole + (1-alpha)*cart
        
        cart_out = pid(pid_input,  cartpole.TAU)  +0.5
        if cart_out > 1:
            cart_out= 1
        elif cart_out< 0:
            cart_out = 0
        
#        print "DIFF CART NET: ",  cart_out - net_out
        
        if (step<000) :
            out = cart_out
        else:
            out = net_out
            print "DIFF: ",  cart_out - net_out

        all_forces.append(out)
        cartpole.doAction(out)
        step += 1
        
#        if step > 1000:
#            break


    all_forces = np.array(all_forces)
    last_W = net.W[net.input_size:, :].ravel().copy()
    print "DIFF_W: ",  first_W - last_W
    print "AVG_DIFF_W: ",  np.mean(np.fabs(first_W - last_W))

    all_obs = np.array(all_obs)

    if markov:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(all_obs[:, 0])
        plt.title("CartPos")

        plt.subplot(2, 2, 3)
        plt.plot(all_obs[:, 1])
        plt.title("CartVel")

        plt.subplot(2, 2, 2)
        plt.plot(all_obs[:, 2])
        plt.title("PoleAngle")

        plt.subplot(2, 2, 4)
        plt.plot(all_obs[:, 3])
        plt.title("PoleVel")

        plt.figure()
        plt.plot(all_forces)
        plt.title("Forces")
    else:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(all_obs[:, 0])
        plt.title("CartPos")

        plt.subplot(3, 1, 2)
        plt.plot(all_obs[:, 1])
        plt.title("PoleAngle")

        plt.subplot(3, 1, 3)
        plt.plot(all_forces[:])
        plt.title("Forces")

    print "\nSteps: ",  step
    print "Avg Steps: ", avg_steps

    plt.show()
