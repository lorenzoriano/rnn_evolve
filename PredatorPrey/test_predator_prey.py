import competitive_scenario as cs
import matplotlib.pyplot as plt
import rnn_evolve.variable_rnn as variable_rnn
import pickle
import numpy as np
import random
from rnn_evolve import kolmogorov

#DIR = "/home/pezzotto/Logs/PredatorPrey/FixedWorld/Hidden3/"
DIR = "/home/pezzotto/Logs/PredatorPrey/FixedWorld/ComplexNet/"
#DIR = "/home/pezzotto/Logs/PredatorPrey/"

pi = np.pi
file = open(DIR + "bestnet.txt", "r")
net = pickle.load(file)
file.close()
print "Net size: ",  net.size
nsensors = 6

dt = 0.1
max_time = 150
prey_speed = 0.4
predator_speed = 0.45
buffer_size=5
lookahead_steps = 10
times = []

all_inputs = []
all_outputs = []

best_session_inputs = []
best_session_outputs = []

for i in xrange(300):
    world = cs.World()
#    world.fixed_distance_top_right(5)
    predator_pos_x = [world.predator[0]]
    predator_pos_y = [world.predator[1]]

    prey_pos_x = [world.prey[0]]
    prey_pos_y = [world.prey[1]]
    elapsed_time = 0

    predator_vw = cs.predator_lookahead_strategy(buffer_size=buffer_size,  lookahead_steps = lookahead_steps)
#    predator_vw = cs.predator_greedy_strategy

    this_session_inputs = []
    this_session_outputs = []

    while not world.collision() and elapsed_time < max_time:
#        v_predator = cs.predator_greedy_strategy(predator_speed,  world)
        v_predator = predator_vw(predator_speed,  world)
        sensor_input = cs.create_prey_sensory_inputs(world,  nsensors)

        input_arr = np.array(sensor_input,  ndmin=2)
        netout = net(input_arr)

#        v = netout[0] * prey_speed
        v = prey_speed
        w = 2.*pi*netout[0, 0] - pi

        v_prey = (v, w)
    #    v_prey = (0.2, 0)
#        v_prey = (0.2, 0.1)
        world.update(v_prey,  v_predator,  dt)
        elapsed_time = elapsed_time + dt

        predator_pos_x.append(world.predator[0])
        predator_pos_y.append(world.predator[1])
        prey_pos_x.append(world.prey[0])
        prey_pos_y.append(world.prey[1])

        all_inputs.append(sensor_input)
        all_outputs.append(v_prey)

        this_session_inputs.append(sensor_input)
        this_session_outputs.append(v_prey)

    if len(times) == 0 or elapsed_time > np.max(times):
        best_predator_x = predator_pos_x
        best_predator_y = predator_pos_y
        best_prey_x = prey_pos_x
        best_prey_y = prey_pos_y
        best_session_inputs = this_session_inputs
        best_session_outputs = this_session_outputs

    times.append(elapsed_time)
    print "Session ",  i,  " time: ",  elapsed_time

print "max: ",  np.max(times),  " min: ",  np.min(times),  " avg: ",  np.mean(times),  " out of ",  len(times)

np.savetxt("/home/pezzotto/tmp/predator_prey_times.txt",  times)

plt.plot([best_predator_x[0]], [best_predator_y[0]], 'bo')
plt.plot(best_predator_x, best_predator_y,  'b')

plt.plot([best_prey_x[0]], [best_prey_y[1]],  'ro')
plt.plot(best_prey_x, best_prey_y,  'r')

elapsed_time = np.max(times)
if elapsed_time < max_time:
    plt.title("Gotcha in "+str(elapsed_time) +" sec")
else:
    plt.title("Missed in steps: " + str(len(best_predator_x)))

best_session_outputs = np.array(best_session_outputs)
#best_session_inputs = np.array(best_session_inputs)
#
#plt.figure()
#plt.plot(best_session_outputs[:, 1])
#plt.title("Prey speed")
#
#plt.figure()
#plt.plot(best_session_inputs[:, 2:])
#plt.title("All the inputs")

kol = np.array(best_session_outputs[:, 1]*100.0,  dtype = np.int)
print "Output Complexity: ",  kolmogorov.kolmogorov_zlib(kol)

np.savetxt(DIR + "best_prey.txt",  np.vstack((best_prey_x, best_prey_y)).T )
np.savetxt(DIR + "best_predator.txt",  np.vstack((best_predator_x, best_predator_y)).T )
np.savetxt(DIR + "times.txt",  times)
np.savetxt(DIR + "outputs.txt",  best_session_outputs)

#np.savetxt(DIR + "inputs.txt",  best_session_inputs)
#np.savetxt(DIR + "all_outputs_random.txt",  all_outputs)

plt.show()
print "done"
