import gym
from gym.envs.box2d.car_racing import CarRacing
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import matplotlib.pyplot as plt
import numpy as np
# import pyglet
# from pyglet import gl
# from pyglet.window import key
# action variables
a = np.array([0.0, 0.0, 0.0])

# init environement
env = CarRacing(render_mode="human")
env.render()
env.reset()

# define variables
total_reward = 0.0
steps = 0
restart = False

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()


while True:
    # perform step
    velocity = env.car.hull.linearVelocity
    s, r, done, tmp, info = env.step(a)
    # speed = info['speed']
    speed = np.sqrt(np.square(velocity[0]) + np.square(velocity[1]))
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # reward
    total_reward += r

    # outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

        # LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        LongC_module.plot_speed(speed, target_speed, steps, fig)

    steps += 1
    env.render()

    # check if stop
    if done or restart or steps >= 600:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        break

env.close()
