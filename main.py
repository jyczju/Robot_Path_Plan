from vision import Vision
from action import Action
from debug import Debugger
from move import Move
import RRT_STAR
import time
import math

if __name__ == '__main__':
	vision = Vision()
	action = Action()
	debugger = Debugger()
	planner = RRT_STAR.RRT_STAR()
	move = Move()
	action.sendCommand(vx=0, vy=0, vw=0)
	# time.sleep(5)
	goal_x, goal_y = -2400, -1500
	while True:
		while vision.my_robot.x < -4500 or vision.my_robot.y < -3000:
			print('wait...')
			time.sleep(0.3)
		start_x, start_y = vision.my_robot.x, vision.my_robot.y
		path_x, path_y= planner.plan(vision, start_x, start_y, goal_x, goal_y)
		print(path_x)
		print(path_y)
		debugger.draw_all(path_x, path_y)
		move.run(action, vision, path_x, path_y)
		if goal_x == -2400:
			goal_x, goal_y = 2400, 1500
		else:
			goal_x, goal_y = -2400, -1500
		# # action.sendCommand(vx=100, vy=0, vw=0)
		# debugger.draw_circle(vision.my_robot.x, vision.my_robot.y)
		
