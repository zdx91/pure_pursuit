#!/usr/bin/env python

import numpy as np
import pygame
from copy import copy, deepcopy
import math


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


class Pose(object):
    """
    Robot pose
    """

    def __init__(self, position = [], heading = 0):
        self.position = np.array(position)
        self.heading = heading


class Robot(object):

    def __init__(
        self,
        initial_pose,
        initial_linear_velocity=0.0,
        initial_angular_velocity=0.0,
        color=(0, 0, 255)
    ):
        """
        Constructor
        :param initial_pose:
        :param initial_linear_velocity:
        :param initial_angular_velocity:
        :param color:
        """

        self.color = color
        self.pose = initial_pose
        self.linear_velocity = initial_linear_velocity
        self.angular_velocity = initial_angular_velocity
        self.trace = [self.pose.position]

    def set_commands(
        self,
        linear_velocity,
        angular_velocity
    ):
        '''
        Set input commands
            linear_velocity: m/s
            angular_velocity: rad/s
        '''

        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    def get_curr_commands(self):
        return self.linear_velocity, self.angular_velocity

    def get_velocity(self):
        """
        Get velocity
        """

        return np.array([np.cos(self.pose.heading), np.sin(self.pose.heading)]) * self.linear_velocity

    def update(
        self,
        dt,
        eps=1e-12
    ):
        x, y, theta = self.pose.position[0], self.pose.position[1], self.pose.heading

        self.trace.append(self.pose.position)

        # update model
        if np.abs(self.angular_velocity) < eps:
            direction = theta + self.angular_velocity * 0.5 * dt
            x += self.linear_velocity * np.cos(direction) * dt
            y += self.linear_velocity * np.sin(direction) * dt
        else:
            old_theta = theta
            radius = self.linear_velocity / self.angular_velocity
            theta = theta + self.angular_velocity * dt
            x += radius * (np.sin(theta) - np.sin(old_theta))
            y -= radius * (np.cos(theta) - np.cos(old_theta))

        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -1.0 * math.pi:
            theta += 2 *math.pi

        self.pose.position = np.array([x, y])
        self.pose.heading = theta

    def draw(self, screen, px2m):
        # draw a triangle representing vehicle
        base_coords = np.array([
            [0.0, -0.05],
            [0.0, 0.05],
            [0.15, 0.0],
            [0.0, -0.05]
        ])

        theta = self.pose.heading
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        coords = np.dot(base_coords, rot.T)
        pygame.draw.lines(
            screen,
            self.color,
            True,
            np.int32((coords + self.pose.position) * px2m)
        )

        # draw trace
        pygame.draw.lines(
            screen,
            [255, 0, 0],
            False,
            np.int32(np.array(self.trace) * px2m)
        )


class World(object):

    def __init__(
        self,
        world_size=[10.0, 10.0],
        start_pose=Pose([1.0, 1.0], 0.0),
        px2m=100
    ):
        self.size = np.array(world_size)
        self.px2m = px2m
        self.start_pose = start_pose
        self.robot = Robot(start_pose)
        self.obstacles = []

    def set_obstacles(self, obstacles):
        # manually set obstacle field
        self.obstacles = obstacles

    def init_screen(self):
        pygame.init()
        screen = pygame.display.set_mode(
            np.int32(self.size * self.px2m)
        )
        pygame.display.set_caption('World')

        screen.fill((255, 255, 255))

        pygame.display.flip()
        return screen

    def in_collision(self):
        vehicle_pos = self.robot.pose.position

        # check world bounds
        if vehicle_pos[0] < 0 \
           or vehicle_pos[1] < 0 \
           or vehicle_pos[0] > self.size[0] \
           or vehicle_pos[1] > self.size[1]:
            return True

        # check obstacle collisions
        for obstacle in self.obstacles:
            if obstacle.in_collision(vehicle_pos):
                return True

        return False

    def update(self, dt):
        self.robot.update(dt)
        for obstacle in self.obstacles:
            obstacle.update(dt)

    def draw(self, screen, waypoints):
        screen.fill((255, 255, 255))
        self.robot.draw(screen, self.px2m)
        for obstacle in self.obstacles:
            obstacle.draw(screen, self.px2m)

        # drawwaypoints
        waypoint_positions = []
        for waypoint in waypoints:
            waypoint_positions.append(waypoint.position)

        # draw waypoints
        if waypoint_positions:
            pygame.draw.lines(
                screen,
                (0, 0, 0),
                False,
                np.int32(np.array(waypoint_positions) * self.px2m,
                width=2)
            )

        pygame.display.flip()

    def get_snapshot(self):
        return deepcopy(self)


if __name__ == "__main__":
    world = World()
    screen = world.init_screen()
