import sys
sys.path.insert(0, "/home/pezzotto/lib/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-i686.egg")
sys.path.insert(0, "..")

import crnn
import pyublas
import crnn.libcartpole as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def gimme_coeffs(forces,  obs):
    o = np.ones(forces.shape)
    A = np.hstack( (forces, o) )
    x ,  _, _, _= lstsq(A, obs)
    m, q = x
    print "M: ",  m,  "Q: ",  q
    
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

if __name__ == '__main__':
    markov = False
    cartpole = cp.SingleCartPole(markov)
    cartpole.FORCE_MAG = 10.0
    cartpole.reset(False)
    pid = PID()

    cartpole.TAU = 0.02
    #cartpole.MASSPOLE = 0.1

    num_trials = 50
    avg_steps = 0
    
    pid = PID()
    obs = []
    acts = []    
    
    for trial in xrange(num_trials):
        cartpole.reset(True)
        cart,  pole = cartpole.getObservation() 
        print "Pole angle: ",  pole
        
        steps = 0    
        while not cartpole.outsideBounds():
            steps += 1
            cart,  pole = cartpole.getObservation() - 0.5
            
            alpha = 0.8
            pid_input = alpha*pole + (1-alpha)*cart

            action = pid(pid_input,  cartpole.TAU) + 0.5
            if action > 1:
                action = 1
            elif action < 0:
                action = 0

            cartpole.doAction(action)
            
            acts.append(action)
            obs.append((cart+ 0.5, pole+0.5))
            
            if steps > 3000:
                break
        avg_steps += steps; 
        print "Steps: ",  steps

    avg_steps = avg_steps / num_trials
    print "Average steps: ",  avg_steps
    obs = np.array(obs)
    
    np.savetxt("/home/pezzotto/tmp/input.txt",  obs)
    acts = np.array(acts,  ndmin=2).T
    np.savetxt("/home/pezzotto/tmp/output.txt",  acts)
    
    gimme_coeffs(acts,  obs[:, 0])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(obs[:, 0])
    plt.title("CartPos, steps = " + str(steps))

    plt.subplot(3, 1, 2)
    plt.plot(obs[:, 1])
    plt.title("PoleAngle")

    plt.subplot(3, 1, 3)
    plt.plot(acts)
    plt.title("Forces")

    plt.figure()
    plt.plot(acts,  obs[:, 0], '.')
    plt.title("Cartpole vs Force")

    plt.show()
    print "Done"
