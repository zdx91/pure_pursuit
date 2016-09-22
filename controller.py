#!/usr/bin/env python

from enum import Enum
import abc
import numpy as np
import logging
from path_planner import PathPlanner


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

    @staticmethod
    def log_list(one_list, list_info):
        logging.debug('--------------Printing {}-----------------'.format(list_info))
        for val in one_list:
            logging.debug(val)


class Util(object):
    """
    Util class
    """

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


class PurePursuit(object):
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
        self.goal_point_moveup_dist = 0.5
        self.line_segment, self.total_path_len = self._parametrize_path()

    def control(self, robot_pose):

        self._update_goal_point(robot_pose)
        goal_point_in_local_frame = self._transform_goal_point_to_robot_frame(robot_pose)
        dist_to_goal_point = np.linalg.norm(goal_point_in_local_frame)

        if dist_to_goal_point < 1e-10:
            # this means we have reached goal, no control needed
            return 0, 0
        else:
            steer = self.desired_linear_velocity * 2.0 * goal_point_in_local_frame[0] / (dist_to_goal_point)
            logging.debug('Planned steer angle is {}'.format(steer))

            return self.desired_linear_velocity, steer

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

        if length_moved > self.total_path_len:
            return self.waypoints[-1].position

        # Since length_moved is guaranteed smaller than self.line_segment[-1], the returned insertio index will not
        # out of range
        line_seg_index = Util.find_insert_place(self.line_segment, length_moved)
        dist_to_end_point_of_line_seg = self.line_segment[line_seg_index] - length_moved

        # unit vector pointing to the start point of the line segment
        end_to_start_vec = self.waypoints[line_seg_index].position - self.waypoints[line_seg_index + 1].position
        unit_vec = (end_to_start_vec) / np.linalg.norm(end_to_start_vec)

        return self.waypoints[line_seg_index + 1].position + dist_to_end_point_of_line_seg * unit_vec

    def _parametrize_path(self):
        """
        Parametrize path
        :return:
        """

        line_segment = []
        curr_dist = 0
        for i in range(len(self.waypoints) - 1):
            curr_dist += np.linalg.norm(self.waypoints[i].position - self.waypoints[i+1].position)
            line_segment.append(curr_dist)

        return line_segment, curr_dist

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
                                    [0,                       0,                       1]])

        coordinate_in_robot_frame = np.dot(np.linalg.inv(rotation_matrix), np.append(goal_point_position_in_global_frame, 1.0))

        return coordinate_in_robot_frame[:2]

if __name__ == '__main__':
    waypoint_list = [[0, 0], [1, 1], [2, 2], [3, 3]]
    waypoints = PathPlanner.create_waypoints(waypoint_list)
    controller = PurePursuit(waypoints, 10, 5, 5)

    LogUtil.set_up_logging('PurePursuit.txt')
    LogUtil.log_list(controller.line_segment, 'dist to line seg')

    logging.info('position is {}'.format(controller.find_position_on_path(np.math.sqrt(3))))
    print(Util.find_insert_place(range(10), 10))