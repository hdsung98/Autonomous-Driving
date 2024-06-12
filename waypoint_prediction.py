import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import sys


def curvature(waypoints):
    waypoint_count = waypoints.shape[1]
    total_curvature = 0
    for index in range(1, waypoint_count - 2):
        vector_a = waypoints[:, index] - waypoints[:, index - 1]
        vector_b = waypoints[:, index + 1] - waypoints[:, index]

        normalized_a = vector_a / np.linalg.norm(vector_a)
        normalized_b = vector_b / np.linalg.norm(vector_b)
        total_curvature += np.dot(normalized_a, normalized_b)

    return total_curvature


def smoothing_objective(waypoints, center_waypoints, curvature_weight=30):
    least_squares_error = np.mean((center_waypoints - waypoints) ** 2)
    calculated_curvature = curvature(waypoints.reshape(2, -1))
    return -1 * curvature_weight * calculated_curvature + least_squares_error


def waypoint_prediction(
    spline_roadside1, spline_roadside2, count_waypoints=6, type_way="smooth"
):
    time_values = np.linspace(0, 1, count_waypoints)
    points_roadside1 = np.array(splev(time_values, spline_roadside1))
    points_roadside2 = np.array(splev(time_values, spline_roadside2))
    predicted_waypoints = (points_roadside1 + points_roadside2) / 2

    if type_way == "center":
        center_waypoints = (points_roadside1 + points_roadside2) / 2
        return center_waypoints.reshape(2, -1)

    elif type_way == "smooth":
        initial_path = predicted_waypoints.flatten()
        smooth_result = minimize(
            smoothing_objective, initial_path, args=(predicted_waypoints.flatten(),)
        )
        path_optimized = smooth_result.x.reshape(2, -1)

        return path_optimized


def target_speed_prediction(
    path_points, total_waypoints=6, max_speed=76, exp_constant=5.3, speed_offset=40
):
    curve = curvature(path_points)
    curv_center = abs(total_waypoints - 3 - curve)
    speed_target = (max_speed - speed_offset) * np.exp(
        -exp_constant * curv_center
    ) + speed_offset
    return speed_target
