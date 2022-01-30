import sys
import socket
import threading

from vision_detection_pb2 import Vision_DetectionFrame, Vision_DetectionRobot

class Vision(object):
	def __init__(self):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.vision_address = '127.0.0.1'
		self.vision_port = 23333
		self.sock.bind((self.vision_address, self.vision_port))
		self.sock.settimeout(1.0)
		self.vision_thread = threading.Thread(target=self.receive_vision)
		self.vision_thread.start()
		self.vision_frame = Vision_DetectionFrame()
		self.blue_robot = [Robot(id=i) for i in range(16)]
		self.yellow_robot = [Robot(id=i) for i in range(16)]
		
	def receive_vision(self):
		while True:
			try:
				data, server = self.sock.recvfrom(65535)
				# print('received message from', server)
				self.vision_frame.ParseFromString(data)
				self.parse_vision()
			except socket.timeout:
				print('VISION TIMED OUT')

	def parse_vision(self):
		# reset visible
		for i in range(16):
			self.blue_robot[i].visible = False
			self.yellow_robot[i].visible = False
		# Store new data
		for robot_blue in self.vision_frame.robots_blue:
			# print('Robot Blue {} pos: {} {}'.format(robot_blue.robot_id, robot_blue.x, robot_blue.y))
			# Store vision info
			self.blue_robot[robot_blue.robot_id].x = robot_blue.x
			self.blue_robot[robot_blue.robot_id].y = robot_blue.y
			self.blue_robot[robot_blue.robot_id].vel_x = robot_blue.vel_x
			self.blue_robot[robot_blue.robot_id].vel_y = robot_blue.vel_y
			self.blue_robot[robot_blue.robot_id].rotate_vel = robot_blue.rotate_vel
			self.blue_robot[robot_blue.robot_id].orientation = robot_blue.orientation
			self.blue_robot[robot_blue.robot_id].visible = True

		for robot_yellow in self.vision_frame.robots_yellow:
			# print('Robot Yellow {} pos: {} {}'.format(robot_yellow.robot_id, robot_yellow.x, robot_yellow.y))
			# Store vision info
			self.yellow_robot[robot_yellow.robot_id].x = robot_yellow.x
			self.yellow_robot[robot_yellow.robot_id].y = robot_yellow.y
			self.yellow_robot[robot_yellow.robot_id].vel_x = robot_yellow.vel_x
			self.yellow_robot[robot_yellow.robot_id].vel_y = robot_yellow.vel_y
			self.yellow_robot[robot_yellow.robot_id].rotate_vel = robot_yellow.rotate_vel
			self.yellow_robot[robot_yellow.robot_id].orientation = robot_yellow.orientation
			self.yellow_robot[robot_yellow.robot_id].visible = True

	@property
	def my_robot(self):
		return self.blue_robot[0]


class Robot(object):
	def __init__(self, id, x=-999999, y=-999999, vel_x=0, vel_y=0, orientation=0, rotate_vel=0, visible=False):
		self.id = id
		self.x = x
		self.y = y
		self.vel_x = vel_x
		self.vel_y = vel_y
		self.orientation = orientation
		self.rotate_vel = rotate_vel
		self.visible = visible


if __name__ == '__main__':
	vision_module = Vision()
