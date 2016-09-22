#!/usr/bin/env python
"""
Use pure pursuit controller to control a differential drive robot to follow a set of waypoints
"""

from world import World, Pose
import numpy as np
from time import sleep
from path_planner import PathPlanner
from controller import PurePursuit

# init world
start_pose = Pose([1.0, 1.0], 0)
world = World(start_pose=start_pose)
# timestep for world update
dt = 0.05
goal_tolerance = 0.25

# init pygame screen for visualization
screen = world.init_screen()

# get vehicle
robot = world.robot

# set vehicle velocity
linear_velocity = 0.5
angular_velocity = 0
robot.set_commands(
    linear_velocity,
    angular_velocity
)

# initialize planner and controller
waypoints, goal = PathPlanner.plan(world, 10)
max_linear_velocity = 1
max_angular_velocity = np.pi / 3.0
look_ahead_dist = 1
controller = PurePursuit(waypoints, max_linear_velocity, max_angular_velocity, look_ahead_dist)

print 'Running dynamic obstacles world'
while True:
    # collision testing
    if world.in_collision():
        print 'Collision'
        break

    # check if we have reached our goal
    vehicle_pose = robot.pose
    goal_distance = np.linalg.norm(vehicle_pose.position - goal)
    if goal_distance < goal_tolerance:
        print 'Goal Reached'
        break

    # if we want to update vehicle commands while running world
    steer = controller.control(vehicle_pose)
    robot.set_commands(linear_velocity, angular_velocity)

    # update world
    world.update(dt)
    # draw world
    world.draw(screen)

# provide some time to view result
sleep(2.0)
