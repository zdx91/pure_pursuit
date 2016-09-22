#!/usr/bin/env python
"""
A simple path planner, get random waypoints in the world
"""

from world import Waypoint
import random
import numpy as np


class PathPlanner(object):
    """
    Path planner
    """

    @staticmethod
    def plan(world, num_waypoints=5):
        if num_waypoints <= 1:
            raise ValueError('invalid parameters for path planner')

        res = []
        for i in range(num_waypoints):
            waypoint = np.array([random.random() * world.size[0], random.random() * world.size[1]])
            res.append(Waypoint(waypoint))

        return res, waypoint

    @staticmethod
    def create_waypoints(waypoint_list):

        res = []
        for waypoint in waypoint_list:
            res.append(Waypoint(waypoint))

        return res