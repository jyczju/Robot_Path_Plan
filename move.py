import time
import math
import numpy as np
from scipy.spatial import KDTree
from action import Action
from vision import Vision
import random


class Move(object):
    # simple:THRE=200, hard:THRE=450
    def __init__(self, MAX_V=3000-500, MAX_A=3000, MAX_W=5, MAX_ALPHA=5, THRE=200, K_RHO=1.2, K_ALPHA=2, K_BETA=0.01):
        self.MAX_V = MAX_V
        self.MAX_A = MAX_A
        self.MAX_W = MAX_W
        self.MAX_ALPHA = MAX_ALPHA
        self.THRE = THRE
        self.K_RHO = K_RHO
        self.K_ALPHA = K_ALPHA
        self.K_BETA = K_BETA
        self.v_now = 0
        self.w_now = 0

    def run(self, action, vision, path_x, path_y):
        for i in range(1, len(path_x)-1):
            while math.hypot(vision.my_robot.x-path_x[i], vision.my_robot.y-path_y[i]) > self.THRE:
                vf, wf = self.trackplan(action, vision, path_x[i], path_y[i])
                print('vf, wf:', vf, wf)
        while math.hypot(vision.my_robot.x-path_x[-1], vision.my_robot.y-path_y[-1]) > 100:
            vf, wf = self.trackplan(action, vision, path_x[-1], path_y[-1])
            print('vf, wf:', vf, wf)

        # 在终点减速到0
        v_now = self.v_now
        w_now = self.w_now
        while v_now > 0.00001 or abs(w_now) > 0.00001:
            v_now = self.v_now
            w_now = self.w_now
            vf = 0
            wf = 0
            v_min = max(-self.MAX_V, v_now-self.MAX_A/30)
            w_max = min(self.MAX_W, w_now+self.MAX_ALPHA/30)
            w_min = max(-self.MAX_W, w_now-self.MAX_ALPHA/30)
            if vf < v_min:
                vf = v_min
            if wf > w_max:
                wf = w_max
            elif wf < w_min:
                wf = w_min
            action.sendCommand(vf, 0, wf)
            print('vf, wf:', vf, wf)
            time.sleep(0.02)
            self.v_now = vf
            self.w_now = wf

    def trackplan(self, action, vision, path_xf, path_yf):
        '''轨迹规划，包括路径规划和避障规划'''
        if self.pathobsfree(vision, self.v_now, self.w_now):
            vf, wf = self.FeedbackControl(vision, path_xf, path_yf)
            vf, wf = self.avoidobs(action, vision, vf, wf)  # hard模式启用
        else:
            vf, wf = self.avoidobs(
                action, vision, self.v_now, self.w_now)  # hard模式启用
        # vf, wf = self.FeedbackControl(vision, path_xf, path_yf)
        # vf, wf = self.avoidobs(action, vision, vf, wf)  # hard模式启用

        if vf == 0:
            vf = 0.00001
        if wf == 0:
            wf = 0.00001
        action.sendCommand(vf, 0, wf)
        time.sleep(0.02)
        self.v_now = vf
        self.w_now = wf
        return vf, wf

    def FeedbackControl(self, vision, path_xf, path_yf):
        '''反馈控制法'''
        x_now = vision.my_robot.x
        y_now = vision.my_robot.y
        w_now = self.w_now
        v_now = self.v_now
        theta = vision.my_robot.orientation
        v_max = min(self.MAX_V, v_now+self.MAX_A/30)
        v_min = max(-self.MAX_V, v_now-self.MAX_A/30)
        w_max = min(self.MAX_W, w_now+self.MAX_ALPHA/30)
        w_min = max(-self.MAX_W, w_now-self.MAX_ALPHA/30)
        dx = path_xf - x_now
        dy = path_yf - y_now
        beta = math.atan2(dy, dx)
        alpha = beta - theta
        while alpha > math.pi:
            alpha -= 2*math.pi
        while alpha < -math.pi:
            alpha += 2*math.pi
        rho = math.hypot(dx, dy)*np.sign(math.cos(alpha))
        if rho > 0:
            vf = self.K_RHO*rho + 300  # 在接近节点时以一个较小的速度前进
        else:
            vf = max(v_min, 0)
        wf = self.K_ALPHA*alpha + self.K_BETA*beta

        if vf > v_max:
            vf = v_max
        elif vf < v_min:
            vf = v_min
        if wf > w_max:
            wf = w_max
        elif wf < w_min:
            wf = w_min
        if -math.pi/2 < alpha < -math.pi/6 or math.pi/6 < alpha < math.pi/2:
            vf = min(vf, 250)  # simple模式
            # vf = min(vf, 360)  # 防止机器人离轨道过远，限定其在alpha角过大时的速度，限制曲率
        return vf, wf

    def avoidobs(self, action, vision, vf, wf):
        '''避障规划'''
        # v_temp = vf
        # w_temp = wf
        v_now = self.v_now
        w_now = self.w_now
        v_max = min(self.MAX_V, v_now+self.MAX_A/30)
        v_min = max(-self.MAX_V, v_now-self.MAX_A/30)
        w_max = min(self.MAX_W, w_now+self.MAX_ALPHA/30)
        w_min = max(-self.MAX_W, w_now-self.MAX_ALPHA/30)
        # 后方有障碍物，加速
        vf = self.backavoidobs(vision, vf, v_now, v_max)
        # 前进路线上有障碍物，动态窗格法介入
        # if not self.pathobsfree(vision, vf, wf):
        while not self.pathobsfree(vision, vf, wf):
            vf, wf = self.dwa(vision, vf, wf, v_max, v_min, w_max, w_min)
            if vf == 0:
                vf = 0.00001
            if wf == 0:
                wf = 0.00001
            action.sendCommand(vf, 0, wf)
            time.sleep(0.02)
            self.v_now = vf
            self.w_now = wf

        return vf, wf

    def dwa(self, vision, vf, wf, v_max, v_min, w_max, w_min):
        '''动态窗格法'''
        eva = -999999
        if abs(self.v_now) < 65:  # 速度小时有可能振荡，要探索跳出振荡的可能
            vlist = np.linspace(max(v_min, -800), v_max, 10)
            wlist = np.linspace(w_min, w_max, 10)
        else:
            vlist = np.linspace(max(v_min, -800), 0.7*v_max+0.3*vf, 10)
            wlist = np.linspace(w_min, 0.7*w_max+0.3*wf, 10)
        for vv in vlist:
            for ww in wlist:
                if not self.pathobsfree(vision, vv, ww):
                    continue
                dis = self.dist(vision, vv, ww)
                # print('dis:',dis)
                if vv <= (2*dis*self.MAX_A)**0.5 and ww <= (2*dis*self.MAX_ALPHA)**0.5:
                    # 评估函数，可能会引起局部振荡
                    eva_temp = dis/650 - 1.5*abs((vv-vf)/max(abs(v_min-vf), abs(v_max-vf))) - 2*abs(
                        (ww-wf)/max(abs(w_max-wf), abs(w_min-wf))) + 0.001*vv/v_max
                    if eva_temp > eva:
                        eva = eva_temp
                        vf = vv
                        wf = ww
        return vf, wf

    def backavoidobs(self, vision, vf, v_now, v_max):
        '''后方避障'''
        if abs(v_now) < 50:
            DIS_BOND = 200
        else:
            DIS_BOND = 250
        # 后方有障碍物时，加速前进 # hard模式启用
        for blue_robot in vision.blue_robot:
            if blue_robot.visible and blue_robot.id > 0 and math.hypot(blue_robot.x-vision.my_robot.x, blue_robot.y-vision.my_robot.y) < DIS_BOND:
                dx_obstacle = blue_robot.x - vision.my_robot.x
                dy_obstacle = blue_robot.y - vision.my_robot.y
                beta_obstacle = math.atan2(dy_obstacle, dx_obstacle)
                alpha_obstacle = beta_obstacle - vision.my_robot.orientation
                while alpha_obstacle > math.pi:
                    alpha_obstacle -= 2*math.pi
                while alpha_obstacle < -math.pi:
                    alpha_obstacle += 2*math.pi
                if abs(alpha_obstacle) >= math.pi*4/5:
                    vf = min(vf+50, v_max, 1800)
        for yellow_robot in vision.yellow_robot:
            if yellow_robot.visible and math.hypot(yellow_robot.x-vision.my_robot.x, yellow_robot.y-vision.my_robot.y) < DIS_BOND:
                dx_obstacle = yellow_robot.x - vision.my_robot.x
                dy_obstacle = yellow_robot.y - vision.my_robot.y
                beta_obstacle = math.atan2(dy_obstacle, dx_obstacle)
                alpha_obstacle = beta_obstacle - vision.my_robot.orientation
                while alpha_obstacle > math.pi:
                    alpha_obstacle -= 2*math.pi
                while alpha_obstacle < -math.pi:
                    alpha_obstacle += 2*math.pi
                if abs(alpha_obstacle) >= math.pi*4/5:
                    vf = min(vf+50, v_max, 1800)
        return vf

    def pathobsfree(self, vision, vf, wf):
        x_now = vision.my_robot.x
        y_now = vision.my_robot.y
        theta = vision.my_robot.orientation
        # print(self.vi)
        # MAX = 90-abs(math.floor(abs(self.vi)**0.33))*5 # hard模式
        if abs(self.v_now) > 2000:
            MAX = 45
            DIS_BOND = 350
        elif abs(self.v_now) > 1000:
            MAX = 40
            DIS_BOND = 250
        else:
            MAX = 35
            DIS_BOND = 250
        # print(MAX)
        dt = np.linspace(1, MAX, MAX)
        dt = dt/20
        for i in range(0, MAX-1):
            if wf == 0:
                xf = x_now + vf*dt[i]*math.cos(theta)
                yf = y_now + vf*dt[i]*math.sin(theta)
                thetaf = theta
            else:
                thetaf = theta + wf*dt[i]
                xf = x_now - vf/wf*math.sin(theta) + vf/wf*math.sin(thetaf)
                yf = y_now + vf/wf*math.cos(theta) - vf/wf*math.cos(thetaf)

            for blue_robot in vision.blue_robot:
                if blue_robot.visible and blue_robot.id > 0 and math.hypot(blue_robot.x-xf, blue_robot.y-yf) < DIS_BOND:
                    return False
            for yellow_robot in vision.yellow_robot:
                if yellow_robot.visible and math.hypot(yellow_robot.x-xf, yellow_robot.y-yf) < DIS_BOND:
                    return False
        return True

    def dist(self, vision, v, w):
        x_now = vision.my_robot.x
        y_now = vision.my_robot.y
        theta = vision.my_robot.orientation
        # print(self.vi)
        # MAX = 90-abs(math.floor(abs(self.vi)**0.33))*5 # hard模式
        MAX = 35
        # print(MAX)
        dis = 999999
        dt = np.linspace(1, MAX, MAX)
        dt = dt/20
        for i in range(0, MAX-1):
            if w == 0:
                xf = x_now + v*dt[i]*math.cos(theta)
                yf = y_now + v*dt[i]*math.sin(theta)
                thetaf = theta
            else:
                thetaf = theta + w*dt[i]
                xf = x_now - v/w*math.sin(theta) + v/w*math.sin(thetaf)
                yf = y_now + v/w*math.cos(theta) - v/w*math.cos(thetaf)

            for blue_robot in vision.blue_robot:
                if blue_robot.visible and blue_robot.id > 0:
                    dis_temp = math.hypot(blue_robot.x-xf, blue_robot.y-yf)
                    if dis_temp < dis:
                        dis = dis_temp
            for yellow_robot in vision.yellow_robot:
                if yellow_robot.visible:
                    dis_temp = math.hypot(yellow_robot.x-xf, yellow_robot.y-yf)
                    if dis_temp < dis:
                        dis = dis_temp
        dis -= 150
        dis += 50
        return dis
