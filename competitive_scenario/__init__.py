import random
import math
import numpy as np

pi = np.pi

class World(object):
    def __init__(self,  desired_dist= None):
        self.randomise_init()
        if desired_dist:
            while np.abs(self.distance() - desired_dist) > 0.5:
                self.randomise_init()
        else:
            while self.distance() < 2:
                self.randomise_init()

        self.prey_speed = (0, 0)
        self.predator_speed=(0, 0)


    def fixed_distance_top_left(self,  distance):
        prey_x = random.uniform(-distance/2.0,  0) #<0
        prey_y = random.uniform(0., distance/2.) #>0

        predator_x = random.uniform(0, distance/2.) #>0

        y2 = prey_y
        x2 = prey_x
        x1 = predator_x
        d = distance

        predator_y = y2 - math.sqrt(-x1 * x1 + 2.0 * x1 * x2 - x2 * x2 + d * d); #<0
        th = math.atan2( prey_y - predator_y,  prey_x - predator_x)

        self.predator = (predator_x,  predator_y,  th)
        self.prey = (prey_x,  prey_y, th + random.uniform(-pi/6.,  pi/6.))

    def fixed_distance_top_right(self,  distance):
        prey_x = random.uniform(0, distance/2.0) #>0
        prey_y = random.uniform(0., distance/2.) #>0

        predator_x = random.uniform(-distance/2., 0) #<0

        y2 = prey_y
        x2 = prey_x
        x1 = predator_x
        d = distance

        predator_y = y2 - math.sqrt(-x1 * x1 + 2.0 * x1 * x2 - x2 * x2 + d * d); #<0
        th = math.atan2( prey_y - predator_y,  prey_x - predator_x)

        self.predator = (predator_x,  predator_y,  th)
        self.prey = (prey_x,  prey_y, th + random.uniform(-pi/6.,  pi/6.))

    def fixed_distance_bottom_left(self,  distance):
        prey_x = random.uniform(-distance/2., 0) #<0
        prey_y = random.uniform(-distance/2., 0) #<0

        predator_x = random.uniform(0, distance/2.) #>0

        y2 = prey_y
        x2 = prey_x
        x1 = predator_x
        d = distance

        predator_y = y2 + math.sqrt(-x1 * x1 + 2.0 * x1 * x2 - x2 * x2 + d * d); #>0
        th = math.atan2( prey_y - predator_y,  prey_x - predator_x)

        self.predator = (predator_x,  predator_y,  th)
        self.prey = (prey_x,  prey_y, th + random.uniform(-pi/6.,  pi/6.))

    def fixed_distance_bottom_right(self,  distance):
        prey_x = random.uniform(0, distance/2.) #>0
        prey_y = random.uniform(-distance/2., 0) #<0

        predator_x = random.uniform(-distance/2., 0) #<0

        y2 = prey_y
        x2 = prey_x
        x1 = predator_x
        d = distance

        predator_y = y2 + math.sqrt(-x1 * x1 + 2.0 * x1 * x2 - x2 * x2 + d * d); #>0
        th = math.atan2( prey_y - predator_y,  prey_x - predator_x)

        self.predator = (predator_x,  predator_y,  th)
        self.prey = (prey_x,  prey_y, th + random.uniform(-pi/6.,  pi/6.))


    def randomise_init(self):
        x = random.uniform(0., 10.)
        y = random.uniform(0, 10.)
        th = random.uniform(-math.pi, math.pi)
        self.prey = (x, y, th)

        x = random.uniform(0., 10.)
        y = random.uniform(0., 10.)
        th = random.uniform(-math.pi, math.pi)
        self.predator = (x, y, th)

    def update(self,  v_prey,  v_predator,  dt):
        self.predator_speed = v_predator
        self.prey_speed = v_prey

        v, w = v_prey
        x, y, th = self.prey
        x = x + v*math.cos(th) * dt
        y = y + v*math.sin(th) * dt
        th = th + w*dt
        if th > pi:
            th = th - 2*pi
        elif th < -pi:
            th = 2*pi + th
        self.prey = (x, y, th)

        v, w = v_predator
        x, y, th = self.predator
        x = x + v*math.cos(th) * dt
        y = y + v*math.sin(th) * dt
        th = th + w*dt
        if th > pi:
            th = th - 2*pi
        elif th < -pi:
            th = 2*pi + th
        self.predator = (x, y, th)

    def distance(self):
        x1, y1, _ = self.prey
        x2, y2, _ = self.predator

        return math.sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) )

    def collision(self):
        return self.distance() < 0.5

def create_prey_sensory_inputs(world,  nsensors):
    prey = world.prey
    predator = world.predator
    d = world.distance()
    ranges = [1.0 for i in xrange(nsensors)]

    if d >= 4.0:
        return ranges

    x1, y1, th = prey
    x2, y2, _ = predator
    angle = math.atan2(y2 - y1,  x2 - x1) - th
    if angle > pi:
        angle = angle - 2*pi
    elif angle < -pi:
        angle = 2*pi + angle

    sensors = np.linspace(-pi,  pi,  nsensors+1)
    for i in xrange(1, len(sensors)):
        if angle >= sensors[i-1] and angle <= sensors[i]:
                ranges[i-1] = d/4.0
                break

    assert np.min(ranges) < 1.0
    return ranges

def create_prey_sensory_inputs_vel(world,  nsensors,  max_prey_speed):
    ranges = create_prey_sensory_inputs(world,  nsensors)

    v, w = world.prey_speed
    v = v/max_prey_speed
    w = (w + pi)/(2.*pi)

    return [v, w] + ranges

def create_predator_sensory_inputs(world):
    x1, y1, _ = world.prey
    x2, y2, th = world.predator
    angle = math.atan2(y1 - y2,  x1 - x2) -th

    if th > pi:
        th = th - 2*pi
    elif th < -pi:
        th = 2*pi + th

    norm_th = (th + pi)/(2.*pi)

    d = world.distance()
    norm_d = d/14.142135623730951
    if norm_d > 1.0:
        norm_d = 1.0

    return [norm_d,  norm_th]

def predator_greedy_strategy(max_v,  world):
    prey = world.prey
    predator = world.predator
    d = world.distance()
#    if d > max_v:
#        v = max_v
#    else:
#        v = d
    v = max_v
    x1, y1, _ = prey
    x2, y2, th = predator
    angle = math.atan2(y1 - y2,  x1 - x2)
    w = angle - th

    if w > pi:
        w = w - 2*pi
    elif w < -pi:
        w = 2*pi + w

    return v, w

from collections import deque

class predator_lookahead_strategy(object):
    def __init__(self, lookahead_steps = 5,  buffer_size = 10,   dt=0.1):
        self.past_prey_x = deque()
        self.past_prey_y = deque()
        self.dt = dt
        self.lookahead = float(buffer_size + lookahead_steps)*dt
        self.buffer_size = buffer_size
        self.old_coeff_x = None
        self.old_coeff_y = None
        self.last_lookahead = 0

    def calculate_next_prey_pos_linear(self):
        x = np.array([i*self.dt  for i in xrange(len(self.past_prey_x))])
        x = np.vstack( (x,np.ones(len(x))) ).T
        y = self.past_prey_x
        coef, _, _, _ = np.linalg.lstsq(x, y)

        next_prey_x = coef[0] * self.lookahead + coef[1]

        y = self.past_prey_y
        coef, _, _, _ = np.linalg.lstsq( x, y)

        next_prey_y = coef[0] * self.lookahead + coef[1]

        return next_prey_x,  next_prey_y

    def calculate_next_prey_pos_quadratic(self):

        if (self.old_coeff_x is None) or (self.last_lookahead >= self.buffer_size):
            #preparing the matrix for LSQ
            lin_x = np.array([i*self.dt  for i in xrange(len(self.past_prey_x))])
            quad_x = np.array([ i*i*self.dt*self.dt  for i in xrange(len(self.past_prey_x))])
            bias_x = np.ones(len(self.past_prey_x))
            x = np.vstack( (quad_x, lin_x,bias_x) ).T

            y = self.past_prey_x
            coef, _, _, _ = np.linalg.lstsq(x, y)
            next_prey_x = coef[0] * self.lookahead * self.lookahead + coef[1] * self.lookahead + coef[2]
            self.old_coeff_x = coef

            y = self.past_prey_y
            coef, _, _, _ = np.linalg.lstsq( x, y)
            next_prey_y = coef[0] * self.lookahead * self.lookahead + coef[1] * self.lookahead + coef[2]
            self.old_coeff_y = coef
            self.last_lookahead = 0

        else:
            next_prey_x = self.old_coeff_x[0] * self.lookahead * self.lookahead + self.old_coeff_x[1] * self.lookahead + self.old_coeff_x[2]
            next_prey_y = self.old_coeff_y[0] * self.lookahead * self.lookahead + self.old_coeff_y[1] * self.lookahead + self.old_coeff_y[2]

        self.last_lookahead = self.last_lookahead + 1
        return next_prey_x,  next_prey_y


    def __call__(self,  max_v,  world):
        prey = world.prey
        predator = world.predator
        d = world.distance()

        v = max_v
        prey_x,  prey_y, _ = prey

        self.past_prey_x.append(prey_x)
        if len(self.past_prey_x) > self.buffer_size:
            self.past_prey_x.popleft()
        self.past_prey_y.append(prey_y)
        if len(self.past_prey_y) > self.buffer_size:
            self.past_prey_y.popleft()

#        next_prey_x,  next_prey_y = self.calculate_next_prey_pos_linear()
        next_prey_x,  next_prey_y = self.calculate_next_prey_pos_quadratic()

        predator_x, predator_y, th = predator
        angle = math.atan2(next_prey_y - predator_y,  next_prey_x - predator_x)
        w = angle - th

        if w > pi:
            w = w - 2*pi
        elif w < -pi:
            w = 2*pi + w

        return v, w

if __name__ == '__main__':

    world = World()

    print "Bottom Left"
    world.fixed_distance_bottom_left(5)
    print "Prey: ",  world.prey
    print  "Predator: ",  world.predator
    print "Distance: ",  world.distance()

    print "done"
