#!/usr/bin/env python

from enum import Enum
import abc
import numpy as np
import logging
from path_planner import PathPlanner
import math


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
        self.goal_point_moveup_dist = 0.01
        self.is_goal_point_reached = False
        self.line_segment, self.total_path_len = self._parametrize_path()

    def control(self, robot):

        robot_pose = robot.pose
        logging.debug('************************************')
        logging.debug('Is robot near goal point? {}'.format(self.is_goal_point_reached))
        if not self.is_goal_point_reached:
            # if goal point not reached, use pure pursuit controller
            self._update_goal_point(robot_pose)
            goal_point_in_local_frame = self._transform_goal_point_to_robot_frame(robot_pose)
            dist_to_goal_point = np.linalg.norm(goal_point_in_local_frame)

            if dist_to_goal_point < 1e-10:
                # this means we have reached goal, no control needed
                logging.debug('Planned control: linear {}, angular {}, robot reaches goal'.format(0, 0))
                return 0, 0
            else:
                steer = self.desired_linear_velocity * 2.0 * goal_point_in_local_frame[0] / (dist_to_goal_point * dist_to_goal_point)
                steer = self.max_angular_velocity if steer > self.max_angular_velocity else steer

                logging.debug('Planned control: linear {}, angular {}, pure pursuit controller'.format(self.desired_linear_velocity, steer))
                return self.desired_linear_velocity, steer
        else:
            # if goal point reached, do not use pure pursuit anymore
            goal_pos = self._find_position_on_path(self.total_path_len)
            robot_position = robot_pose.position
            theta = robot_pose.heading

            vec_to_goal = goal_pos - robot_position
            vec_to_goal_angle = math.atan2(vec_to_goal[1], vec_to_goal[0])

            angle_diff = vec_to_goal_angle - theta
            if abs(angle_diff) > 1e-10:
                if abs(angle_diff) > np.pi:
                    angle_diff = (-1.0) * angle_diff / abs(angle_diff) * (2.0 * math.pi - abs(angle_diff))

                circular_move_radius = np.linalg.norm(vec_to_goal) / 2.0 / np.cos(abs(abs(angle_diff) - math.pi / 2.0))
                planned_steer = self.desired_linear_velocity / circular_move_radius * (angle_diff / abs(angle_diff))
                planned_steer = self.max_angular_velocity if planned_steer > self.max_angular_velocity else planned_steer

                logging.debug('Planned control: linear {}, angular {}, robot in neighborhood of goal'.format(self.desired_linear_velocity, planned_steer))
                return self.desired_linear_velocity, planned_steer
            else:
                # current robot heading is towards goal postion, move to there directly
                logging.debug('Planned control: linear {}, angular {}, robot heading points towards goal'.format(self.desired_linear_velocity, 0))
                return self.desired_linear_velocity, 0

    def _update_goal_point(self, robot_pose):
        """
        Find goal point on the path
        :param robot_pose:
        :return: true if goal point reaches end of the path
        """

        # start searching the goal point from the nearest point on the path to the current robot position
        goal_point_search_s_coordinate = self._find_nearest_path_point(robot_pose)
        goal_point_search_position = self._find_position_on_path(goal_point_search_s_coordinate)
        logging.debug('nearest point on the path to robot: {}'.format(goal_point_search_position))

        while np.linalg.norm(goal_point_search_position - robot_pose.position) < self.look_ahead_distance:
            if goal_point_search_s_coordinate + self.goal_point_moveup_dist > self.total_path_len:
                break
            else:
                goal_point_search_s_coordinate += self.goal_point_moveup_dist
                goal_point_search_position = self._find_position_on_path(goal_point_search_s_coordinate)

        self.goal_point = goal_point_search_s_coordinate

        if abs(self.goal_point - self.total_path_len) <= 1e-1:
            self.is_goal_point_reached = True

        logging.debug('Updated goal point: {}, dist between goal point and robot position: {}, controller '
                      'look_ahead_distance: {}'.format(goal_point_search_position, np.linalg.norm(goal_point_search_position - robot_pose.position), self.look_ahead_distance))

    def _find_nearest_path_point(self, robot_pose):
        """
        Find the nearest path point to the robot
        """

        """Parametrize the line by p0 = t*p1 + (1-t)p2, and let p3 = robot_pose.position, then (p3 - p0) is perpendicular
           to (p2-p1)
        """
        nearest_dist = float('inf')

        # this is the s coordinate of the nearest point, we parametrize any point on a path by f(s), s is the length
        # along the path to the starting point
        nearest_point_s_coordinate = 0

        # the distance from the starting point to the curr line segment, that is, the total length of the previous line
        # segments
        dist_to_curr_line_segment = 0
        for index, start_waypoint in enumerate(self.waypoints):
            if index < len(self.waypoints) - 1:
                end_waypoint = self.waypoints[index + 1]
                line_len = np.linalg.norm(end_waypoint.position - start_waypoint.position)

                nearest_dist_to_curr_line = np.linalg.norm(robot_pose.position - start_waypoint.position)

                # s coordinate of the nearest point on the current line segment to the robot position
                curr_line_segment_nearest_point_s_coordinate = dist_to_curr_line_segment
                if np.linalg.norm(robot_pose.position - end_waypoint.position) < nearest_dist_to_curr_line:
                    nearest_dist_to_curr_line = np.linalg.norm(robot_pose.position - end_waypoint.position)
                    curr_line_segment_nearest_point_s_coordinate = dist_to_curr_line_segment + line_len

                if line_len > 1e-10:
                    t = (-1.0) * np.dot(robot_pose.position - start_waypoint.position,
                                 start_waypoint.position - end_waypoint.position) / (line_len * line_len)
                    if 0 < t < 1:
                        perpendicular_point = t * end_waypoint.position + (1 - t) * start_waypoint.position
                        if np.linalg.norm(robot_pose.position - perpendicular_point) < nearest_dist_to_curr_line:
                            nearest_dist_to_curr_line = np.linalg.norm(robot_pose.position - perpendicular_point)
                            curr_line_segment_nearest_point_s_coordinate = dist_to_curr_line_segment + \
                                                                         np.linalg.norm(start_waypoint.position - perpendicular_point)

                if nearest_dist_to_curr_line < nearest_dist:
                    nearest_dist = nearest_dist_to_curr_line
                    nearest_point_s_coordinate = curr_line_segment_nearest_point_s_coordinate

                dist_to_curr_line_segment += line_len

        return nearest_point_s_coordinate

    def _find_position_on_path(self, length_moved):
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

        goal_point_position_in_global_frame = self._find_position_on_path(self.goal_point)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), robot_position[0]],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), robot_position[1]],
                                    [0,                       0,                       1]])

        coordinate_in_robot_frame = np.dot(np.linalg.inv(rotation_matrix), np.append(goal_point_position_in_global_frame, 1.0))

        return coordinate_in_robot_frame[:2]

if __name__ == '__main__':
    waypoint_list = [[0, 0], [1, 1], [2, 2], [3, 3]]
    waypoints, goal = PathPlanner.create_waypoints(waypoint_list)
    controller = PurePursuit(waypoints, 10, 5, 5)

    print('Test find position based on s coordinate, position is {}'.format(controller._find_position_on_path(np.math.sqrt(3))))
    print('Test binary search insertion place function: insertion place is {}'.format(Util.find_insert_place(range(10), 10)))