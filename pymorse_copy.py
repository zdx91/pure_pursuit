import json
from time import sleep

import numpy as np

from controller import LogUtil, PurePursuit
from path_planner import PathPlanner
from world import Pose, Robot, World

# Read in waypoints
def get_waypoint_list():

    with open('path.json', 'r') as f_in:
        path = json.load(f_in)
    waypoint_list = []
    for point in path:
        waypoint_list.append([point['x'] + 100., point['y'] + 40.])
    return waypoint_list

# Call pure pursuit controller
waypoint_list = get_waypoint_list()
waypoints, goal = PathPlanner.create_waypoints(waypoint_list)
world = World()
world.robot = Robot(Pose(waypoints[0].position, np.pi))

max_linear_velocity = 6
max_angular_velocity = np.pi / 6.0
look_ahead_dist = 0.5
controller = PurePursuit(waypoints, max_linear_velocity, max_angular_velocity, look_ahead_dist)

# init pygame screen for visualization
screen = world.init_screen()
goal_tolerance = 3
dt = 0.1
while True:

    # check if we have reached our goal
    vehicle_pose = world.robot.pose
    goal_distance = np.linalg.norm(vehicle_pose.position - goal)
    if goal_distance < goal_tolerance:
        print('Goal Reached')
        break

    # if we want to update vehicle commands while running world
    planned_linear_velocity, steer = controller.control(world.robot)
    world.robot.set_commands(planned_linear_velocity, steer)

    # update world
    world.update(dt)
    # draw world
    world.draw(screen, waypoints)
    print(world.robot.pose.position[0] - 100., world.robot.pose.position[1] - 40)

# pause some time to view result
sleep(1.0)


# Append a vw actuator to the robot, and set linear velocity

# Get robot current pose

# Input the current pose into the pure pursuit controller to get the angular velocity
# Update the angular velocity of the appended vw acutuator 
# Run the Morse simulator
