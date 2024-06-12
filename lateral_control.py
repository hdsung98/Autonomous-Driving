import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

eps = 1e-6

class LateralController:
    """
    Lateral control using the Stanley controller

    functions:
        stanley

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    """

    def __init__(self, gain_constant=0.9, damping_constant=0.00005):
        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0

    def stanley(self, waypoints, speed):

        gain_const = self.gain_constant
        

        crosstrack_err = waypoints[0, 0] - 48
        main_err = np.arctan(
            (waypoints[0, 1] - waypoints[0, 0]) / (waypoints[1, 1] - waypoints[1, 0])
        )

        steering_angle = main_err + np.arctan(
            (gain_const * crosstrack_err) / (speed + eps)
        )

        steering_angle = steering_angle - self.damping_constant * (
            steering_angle - self.previous_steering_angle
        )

        self.previous_steering_angle = steering_angle

        steering_angle = np.clip(steering_angle, -0.4, 0.4)
        steering_angle = steering_angle / 0.4

        return steering_angle
