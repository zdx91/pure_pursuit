#!/usr/bin/env python

from enum import Enum
import abc
import numpy as np


class Pose(object):
    """
    Robot pose
    """

    def __init__(self, position = [], heading = 0):
        self.position = np.array(position)
        self.heading = heading


class Waypoint(object):
    """
    Waypoint
    """

    def __init__(self, position = [], heading = 0, curvature = 0):
        self.position = np.array(position)
        self.heading = heading
        self.curvature = curvature


class ControllerType(Enum):
    pure_pursuit = 1


class ControllerFactory(object):
    """
    Create a controller
    """

    @staticmethod
    def create_controller(controller_type):
        """

        :param controller_type:
        :type: ControllerType
        :return:
        """

        if controller_type == ControllerType.pure_pursuit:
            return ControllerPurePursuit()


class Controller(object):
    """
    Controller base
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def control(self, robot_pose):
        raise NotImplementedError


class ControllerPurePursuit(Controller):
    """
    Pure pursuit controller
    """

    def __init__(self, waypoints = [], linear_velocity = 0, max_angular_velocity = 0, look_ahead_distance = 0):
        self.waypoints = waypoints
        self.desired_linear_velocity = linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.look_ahead_distance = look_ahead_distance

    def control(self, robot_pose):
        pass

    def _find_closest_waypoint(self, robot_pose):
        """
        Find closet waypoint to the current robot pose
        :param robot_pose:
        :type: Pose
        :return:
        """

        closest_dist = float('inf')
        closest_waypoint = None
        closest_waypoint_index = 0
        for index, waypoint in enumerate(self.waypoints):
            dist = np.norm(robot_pose.position - waypoint.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_waypoint = waypoint
                closest_waypoint_index = index

        return closest_waypoint_index, closest_waypoint

    def _find_nearest_path_point(self, start_waypoint, end_waypoint, robot_pose):
        """
        Find the nearest path point to the robot
        :param start_waypoint:
        :param end_waypoint:
        :param robot_pose:
        :return:
        """

        """Parametrize the line by p0 = t*p1 + (1-t)p2, and let p3 = robot_pose.position, then (p3 - p0) is perpendicular
           to (p2-p1)
        """
        t = -1.0 * np.dot(robot_pose.position - start_waypoint.position, start_waypoint.position - end_waypoint.position)\
            / np.norm(end_waypoint.position - start_waypoint.position)
        nearest_path_point = t * end_waypoint.position + (1 - t) * start_waypoint.position

        return t, nearest_path_point

    def _find_goal_point(self, robot_pose):
        """
        Find goal point on the path
        :param robot_pose:
        :return:
        """

        # parameter is the nearest path point parameter on the line
        parameter = 0
        nearest_dist = float('inf')
        candidate_line_seg_index = 0
        for index, waypoint in enumerate(self.waypoints):
            if index < len(self.waypoints) -1:
                t, dist = self._find_nearest_path_point(waypoint, self.waypoints[index + 1], robot_pose)
                if dist < nearest_dist:
                    parameter = t
                    candidate_line_seg_index = index
                    nearest_dist = dist

        # parameter greater than 1 means the robot has passed the endpoint of the current line segment, so we need
        # to move to the next line segment
        if parameter >= 1:
            t, dist = self._find_nearest_path_point(waypoint, self.waypoints[index + 1], robot_pose)

        closest_waypoint_index, closest_waypoint = self._find_closest_waypoint(robot_pose)
        previous_waypoint = None
        next_waypoint = None
        if 0 <= closest_waypoint_index - 1 < len(self.waypoints):
            previous_waypoint = self.waypoints[closest_waypoint_index - 1]

        if 0 <= closest_waypoint_index + 1 < len(self.waypoints):
            next_waypoint = self.waypoints[closest_waypoint_index + 1]

        if previous_waypoint is not None:

