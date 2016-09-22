#!/usr/bin/env python

from enum import Enum
import abc
import numpy as np
import logging
from collections import OrderedDict


class LogUtil(object):
    """
    Log util
    """

    @staticmethod
    def set_up_logging(log_file_name, logging_level=logging.DEBUG):
        logging.basicConfig(filename=log_file_name, filemode='w', level=logging_level,
                            format='%(asctime)s %(message)s')
        # logging.getLogger().addHandler(logging.StreamHandler())

    @staticmethod
    def log_dict(one_dict, dict_info):
        logging.debug('--------------Printing {}-----------------'.format(dict_info))
        for k in one_dict:
            logging.debug('key: {}, value: {}'.format(k, one_dict[k]))


class Util(object):

    @staticmethod
    def create_waypoints(waypoints_list):
        """

        :param waypoints_list: a list with each element being a list containing 2 elements
        :return:
        """
        res = []
        for index, waypoint in enumerate(waypoints_list):
            res.append(Waypoint(waypoint))

        return res

    @staticmethod
    def find_insert_place(sorted_list, val):
        """
        Use binary search to find the place where the val should be inserted
        :param sorted_list:
        :param val:
        :return:
        """

        start, middle, end = 0, 0, len(sorted_list) -1
        while start <= end:
            middle = start + (end - start) / 2
            if val == sorted_list[middle]:
                return middle
            if val < sorted_list[middle]:
                end = middle - 1
            else:
                start = middle + 1

        return start


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

    def __str__(self):
        return 'Position is {}, heading is {}'.format(self.position, self.heading)


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

    def __init__(self, waypoints, linear_velocity, max_angular_velocity, look_ahead_distance):
        """
        :param waypoints: a list of Waypoints
        :param linear_velocity:
        :param max_angular_velocity:
        :param look_ahead_distance:
        """
        if not waypoints or linear_velocity <= 0 or max_angular_velocity <=0 or look_ahead_distance <= 0:
            raise ValueError('waypoints can not be null, and all other parameter need to be positive!')

        self.waypoints = waypoints
        self.desired_linear_velocity = linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.look_ahead_distance = look_ahead_distance
        # goal_point is in 1D along the path, parametrized by the distance to the first waypoint
        self.goal_point = 0
        self.goal_point_moveup_dist = 2
        self.sorted_dist_line_segment_index, self.total_path_len = self._create_line_length_to_segment_index()

    def control(self, robot_pose):
        pass

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
            / np.linalg.norm(end_waypoint.position - start_waypoint.position)
        nearest_path_point = t * end_waypoint.position + (1 - t) * start_waypoint.position

        return t, nearest_path_point

    def _update_goal_point(self, robot_pose):
        """
        Find goal point on the path
        :param robot_pose:
        :return:
        """

        while np.linalg.norm(self.find_position_on_path(self.goal_point) - robot_pose.position) < \
            self.look_ahead_distance or self.goal_point < self.total_path_len:
            self.goal_point += self.goal_point_moveup_dist

    def find_position_on_path(self, length_moved):
        """
        Parametrize the path using the length moved along the path
        :return:
        """

        if len(self.waypoints) == 1:
            return self.waypoints[0].position

        line_seg_index = 0
        dist_to_end_point_of_line_seg = 0
        for dist, index in self.sorted_dist_line_segment_index.items():
            if dist > length_moved:
                line_seg_index = index
                dist_to_end_point_of_line_seg = dist - length_moved
                break

        # unit vector pointing to the start point of the line segment
        end_to_start_vec = self.waypoints[line_seg_index].position - self.waypoints[line_seg_index + 1].position
        unit_vec = (end_to_start_vec) / np.linalg.norm(end_to_start_vec)

        return self.waypoints[line_seg_index + 1].position + dist_to_end_point_of_line_seg * unit_vec

    def _create_line_length_to_segment_index(self):
        """
        Create moved dist to line segment index
        :return:
        """

        dist_to_index_map = {}
        curr_dist = 0
        for i in range(len(self.waypoints) - 1):
            curr_dist += np.linalg.norm(self.waypoints[i].position - self.waypoints[i+1].position)
            dist_to_index_map[curr_dist] = i

        return OrderedDict(sorted(dist_to_index_map.items(), key=lambda x:x[0])), curr_dist

    def _transform_goal_point_to_robot_frame(self, robot_pose):
        """
        Transform the goal point coordinate from global frame into local frame
        :param robot_pose:
        :return:
        """

        rotation_angle = robot_pose.heading
        robot_position = robot_pose.position

        goal_point_position_in_global_frame = self.find_position_on_path(self.goal_point)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), robot_position[0]],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), robot_position[1]],
                                    0,                       0,                       1])

        return np.linalg.inv(rotation_matrix) * np.append(goal_point_position_in_global_frame, 1.0)


if __name__ == '__main__':
    waypoint_list = [[0, 0], [1, 1], [2, 2], [3, 3]]
    waypoints = Util.create_waypoints(waypoint_list)
    controller = ControllerPurePursuit(waypoints, 10, 5, 5)

    LogUtil.set_up_logging('PurePursuit.txt')
    LogUtil.log_dict(controller.sorted_dist_line_segment_index, 'dist to line seg')

    logging.info('position is {}'.format(controller.find_position_on_path(np.math.sqrt(0))))
    print(Util.find_insert_place(range(10), 10))