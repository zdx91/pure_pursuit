#!/usr/bin/env python
"""
Use pure pursuit controller to control a differential drive robot to follow a set of waypoints
"""

from world import World, Pose, Robot
import numpy as np
from time import sleep
from path_planner import PathPlanner
from controller import PurePursuit, LogUtil

# setup logging
LogUtil.set_up_logging('PurePursuit.txt')

# init world
world = World([15, 15])
# timestep for world update
dt = 0.1
goal_tolerance = 0.25

# initialize planner and controller
# waypoints, goal = PathPlanner.plan(world, 10)
waypoint_list = [[1.5, 3], [2, 6], [3, 7], [4, 6], [5, 6.5], [6, 8], [8, 8], [9, 6], [6, 4]]
waypoints, goal = PathPlanner.create_waypoints(waypoint_list)
world.robot = Robot(Pose(waypoints[0].position, np.pi))

max_linear_velocity = 1
max_angular_velocity = np.pi / 3.0
look_ahead_dist = 0.1
controller = PurePursuit(waypoints, max_linear_velocity, max_angular_velocity, look_ahead_dist)

# init pygame screen for visualization
screen = world.init_screen()

print 'Running dynamic obstacles world'
while True:
    # collision testing
    if world.in_collision():
        print 'Collision'
        break

    # check if we have reached our goal
    vehicle_pose = world.robot.pose
    goal_distance = np.linalg.norm(vehicle_pose.position - goal)
    if goal_distance < goal_tolerance:
        print 'Goal Reached'
        break

    # if we want to update vehicle commands while running world
    planned_linear_velocity, steer = controller.control(world.robot)
    world.robot.set_commands(planned_linear_velocity, steer)

    # update world
    world.update(dt)
    # draw world
    world.draw(screen, waypoints)

# provide some time to view result
sleep(2.0)
