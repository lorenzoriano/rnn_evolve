import competitive_scenario as cs
import matplotlib.pyplot as plt
import rnn_evolve.variable_rnn as variable_rnn
import pickle
import numpy as np
import random
from rnn_evolve import kolmogorov
import time
import sys

start_time = time.time()

DIR = "/home/pezzotto/Logs/PredatorPrey/FixedWorld/Selector5Buffer/"

pi = np.pi
file = open(DIR + "bestnet.txt", "r")
net = pickle.load(file)
file.close()
#print "RANDOM NETWORK"
#net = variable_rnn.SelectorRandomRNN(net)
print "Net size: ",  net.size
print "Net class: ",  net.__class__
nsensors = 6
#net.to_dot("/home/pezzotto/tmp/net.dot")
#sys.exit(0)


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

best_session_weights = []
best_starting_distance = 0

best_prey_in_sight = []

all_distances = []
all_times = []

for i in xrange(5):
    world = cs.World()
#    world.fixed_distance_top_left(5)
    predator_pos_x = [world.predator[0]]
    predator_pos_y = [world.predator[1]]

    prey_pos_x = [world.prey[0]]
    prey_pos_y = [world.prey[1]]
    elapsed_time = 0

    predator_vw = cs.predator_lookahead_strategy(buffer_size=buffer_size,  lookahead_steps = lookahead_steps)

    this_session_inputs = []
    this_session_outputs = []
    this_session_weights = []
    starting_distance = world.distance()

    prey_in_sight = []
    while (not world.collision()) and (elapsed_time < max_time):
        v_predator = predator_vw(predator_speed,  world)
        sensor_input = cs.create_prey_sensory_inputs(world,  nsensors)

        if np.any(np.array(sensor_input) != 1.0):
            prey_in_sight.append(1)
        else:
            prey_in_sight.append(0)

        input_arr = np.array(sensor_input)
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
        this_session_weights.append(net.get_weights())


    if len(times) == 0 or elapsed_time > np.max(times):
        best_predator_x = predator_pos_x
        best_predator_y = predator_pos_y
        best_prey_x = prey_pos_x
        best_prey_y = prey_pos_y
        best_session_inputs = this_session_inputs
        best_session_outputs = this_session_outputs
        best_session_weights = this_session_weights
        best_starting_distance = starting_distance
        best_prey_in_sight = prey_in_sight

    all_times.append(elapsed_time)
    all_distances.append(starting_distance)
    times.append(elapsed_time)
    print "Session ",  i,  " time: ",  elapsed_time


print "max: ",  np.max(times),  " min: ",  np.min(times),  " avg: ",  np.mean(times),  " out of ",  len(times),  " data"

### Plotting stuff

#np.savetxt(DIR+"selector_network_times.txt",  all_times)
plt.figure();
plt.hist(all_times,  20)

time_axis = np.array(range(1, len(best_predator_x)),  dtype = np.float32) * dt
plt.figure()
plt.plot(all_distances, all_times, 'o')
plt.title("Best time "+str(np.max(times)) + " with starting distance "  + str(best_starting_distance))

plt.figure()
plt.plot(time_axis,  best_prey_in_sight, 'o')
plt.title("Prey in sight")

plt.figure()
plt.subplot(2, 2, 1)
plt.plot([best_predator_x[0]], [best_predator_y[0]], 'bo')
plt.plot(best_predator_x, best_predator_y,  'b')

#plt.plot([best_prey_x[0]], [best_prey_y[1]],  'ro')
plt.plot(best_prey_x, best_prey_y, 'ro')
plt.plot(best_prey_x, best_prey_y,  'r')

elapsed_time = np.max(times)
if elapsed_time < max_time:
    plt.title("Gotcha in "+str(elapsed_time) +" sec")
else:
    plt.title("Missed in " + str(elapsed_time)+" sec")

best_session_outputs = np.array(best_session_outputs)
best_session_inputs = np.array(best_session_inputs)

plt.subplot(2, 2, 2)
plt.plot(time_axis, best_session_outputs[:, 1])
plt.title("Prey speed")

plt.subplot(2, 2, 3)
plt.plot(time_axis, best_session_inputs)
plt.title("All the inputs")

plt.subplot(2, 2, 4)
plt.plot(time_axis, np.array(best_session_weights)[:, 0])
plt.title("Weights for angular velocity")


kol = np.array(best_session_outputs[:, 1]*100.0,  dtype = np.int)
print "Output Complexity: ",  kolmogorov.kolmogorov_zlib(kol)

#print "CHANGING FOR RANDOM NETOWRK"
#DIR = "/home/pezzotto/Logs/PredatorPrey/FixedWorld/RandomNetwork/"
np.savetxt(DIR + "best_prey.txt",  np.vstack((best_prey_x, best_prey_y)).T )
np.savetxt(DIR + "best_predator.txt",  np.vstack((best_predator_x, best_predator_y)).T )
np.savetxt(DIR + "times.txt",  times)

#np.savetxt("/home/pezzotto/tmp/"+"all_inputs.txt",  best_session_inputs)
#np.savetxt(DIR + "all_outputs_random.txt",  all_outputs)

print "Simulation time: ",  time.time() -start_time,  "s"
plt.show()
print "done"

