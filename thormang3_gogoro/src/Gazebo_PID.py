#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64,Float32MultiArray,Bool

from time import sleep,time
import numpy as np
import sys
import os
import curses

# from PCL_GRAB import grab_position
from gogoro_env import GogoroEnv
from utils import *

def log(tag, msg):
    rospy.loginfo(tag + ': ' + msg)
    
class PID():
    def __init__(self):
        ## Balance ##
        self.P_gain = 2
        self.I_gain = 0.001
        self.D_gain = 4
        
        self.last_error = None
        self.i_accum = 0
        self.refer = np.radians(0)
        
        ## Tile ##
        self.tile_P = 0.2
        self.tile_I = 0.001
        self.tile_D = 0.25
        
        # self.tile_P = 1
        # self.tile_I = 0.002
        # self.tile_D = 0.8

        self.tile_refer = np.radians(0)
        self.tile_last_error = None
        self.tile_i_accum = 0

        self.roll_state = 0
        self.pitch_state = 0
        self.yaw_state = 0
        
        self.qua = [0,0,0,0]
        
        self.angvel_x = 0
        self.angvel_y = 0
        self.angvel_z = 0
        
        self.linear_x = 0
        self.linear_y = 0
        self.linear_z = 0
        
        self.action_angle = 0
        
        self.speed_x =  0 
        
        self.accum_ang = 0

        self.time_ratio = 0.5 # depends on simulation time ratio
        
        self.break_signal = False
        
        self.data_pub = rospy.Publisher('/thormang3_gogoro/steering/data', Float32MultiArray, queue_size = 5)
        
        rospy.Subscriber("/thormang3_gogoro/steering/break", 
                         Bool, self._break_callback)
        
        self.imu_sub = rospy.Subscriber("/robotis/sensor/imu", Imu, self._imu_cb)
        
    def controller(self,state):
        
        ## Tile ##
        tile_error = -(self.yaw_state - self.tile_refer) # rad/s
        
        self.tile_i_accum += tile_error
        
        if self.tile_last_error == None:
            tile_error_dot = 0
        else:
            tile_error_dot = tile_error - self.tile_last_error
            
        tile_P_signal = self.tile_P * tile_error #/ P_offset
        tile_I_signal = self.tile_I * self.tile_i_accum
        tile_D_signal = self.tile_D * tile_error_dot
        
        self.tile_last_error = tile_error
        
        tile_angle    = tile_P_signal + tile_I_signal + tile_D_signal
        
        ## Balance ##
        error = (state - self.refer)
        
        self.i_accum += error
        
        if self.last_error == None:
            error_dot = 0
        else:
            error_dot = error - self.last_error
        
        P_scale = 12
        if abs(self.speed_x) < 10 :
            P_offset = 9
        else:
            P_offset = 2 * abs(self.speed_x)
        
        P_signal = self.P_gain*error #(self.P_gain * P_scale/P_offset )* error 
        I_signal = self.I_gain * self.i_accum
        D_signal = self.D_gain * error_dot
        
        angle    = P_signal + I_signal + D_signal
        
        self.last_error = error 


        
        
        angle = angle #+ tile_angle
        
        ## Limit ##            
        if angle >= np.radians(14):
            angle = np.radians(14)
        elif angle <= np.radians(-14):
            angle = np.radians(-14)
        
        
        self.action_angle = np.degrees(angle)
        
        # print("--------------------")

        # print("P_signal : ",np.degrees(P_signal))
        # print("I_signal : ",np.degrees(I_signal))
        # print("D_signal : ",np.degrees(D_signal))
        # print("")
        
        # print("Tile_P_signal : ",np.degrees(tile_P_signal))
        # print("Tile_I_signal : ",np.degrees(tile_I_signal))
        # print("Tile_D_signal : ",np.degrees(tile_D_signal))
        # print("")

        # print("speed_x (km/h) : ",3.6*0.25*self.speed_x) #m/s to km/h and the wheel radius is 0.25  
        
        gogoro_data = Float32MultiArray()
        gogoro_data.data = [self.speed_x,self.angvel_z,np.degrees(self.roll_state),np.degrees(self.yaw_state),np.degrees(angle)]
        self.data_pub.publish(gogoro_data)
        
        
        # print("x : ",self.qua[0])
        # print("y : ",self.qua[1])
        # print("z : ",self.qua[2])
        # print("w : ",self.qua[3])
        
        # print("roll_state (deg) :",np.degrees(self.roll_state))
        # print("pitch_state (deg) :",np.degrees(self.pitch_state))
        # print("yaw_state (deg) : ",np.degrees(self.yaw_state))
        # print("")
        
        # print("angvel_x (rad) : ",self.angvel_x)
        # print("angvel_y (rad) : ",self.angvel_y)
        # print("angvel_z (rad) : ",self.angvel_z)
        # print("accum : ",self.accum_ang)
        # print("")

        # print("angvel_x (deg) : ",np.degrees(self.angvel_x))
        # print("angvel_y (deg) : ",np.degrees(self.angvel_y))
        # print("angvel_z (deg) : ",np.degrees(self.angvel_z))
        # print("accum (deg) : ",np.degrees(self.accum_ang))
        # print("")
        
        # print("linear_x : ",self.linear_x)
        # print("linear_y : ",self.linear_y)
        # print("linear_z : ",self.linear_z)
        # print("")

        # print("P_gain : ",(self.P_gain * P_scale/P_offset ))
        # print("speed_x (m/s): ",self.speed_x ) 
        # print("gogoro roll (deg) : ", np.degrees(self.roll_state))
        # print("steering angle (deg) : ", self.action_angle )
        # print("")

        # print("tile angle (deg) :",np.degrees(tile_angle))
        # print("--------------------\n")        
        
        return angle
    
    def _imu_cb(self,msg):
        quaternion = [msg.orientation.x, 
                      msg.orientation.y, 
                      msg.orientation.z,
                      msg.orientation.w]
        self.qua = quaternion
        euler = quaternion_to_euler(*quaternion)

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2] 
        
        self.angvel_x = msg.angular_velocity.x
        self.angvel_y = msg.angular_velocity.y
        self.angvel_z = msg.angular_velocity.z
        
        self.linear_x = msg.linear_acceleration.x
        self.linear_y = msg.linear_acceleration.y
        self.linear_z = msg.linear_acceleration.z

        self.roll_state = roll + np.radians(3)
        self.pitch_state = pitch
        self.yaw_state = yaw #+ np.radians(3)
         
        # if abs(self.linear_x) > 0.4:
        self.speed_x += (self.linear_x * self.time_ratio)
            
        self.accum_ang += self.angvel_y * self.time_ratio
        
        
        if abs(self.roll_state > np.radians(50)):
            self.reset()
    
    def _break_callback(self,msg):
        self.break_signal = msg.data
    
    def reset(self):
        sleep(0.2)
        
        self.roll_state = 0
        self.pitch_state = 0
        self.yaw_state = 0
        
        self.angvel_x = 0
        self.angvel_y = 0
        self.angvel_z = 0
        
        self.linear_x = 0
        self.linear_y = 0
        self.linear_z = 0
        
        self.action_angle = 0
        
        self.speed_x =  0
        
        self.accum_ang = 0
        
        self.last_error = None
        self.i_accum = 0
        
        self.tile_last_error = None
        self.tile_i_accum = 0

    
if __name__ == "__main__":
    _node_name = 'PID_Agent'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))
    
    ## Test ##
    env = GogoroEnv(action_scale=1,speed=20) # speed unit is rad/s , rad/s to km/h -> 3.6*r*w = 3.6*0.25*speed = 0.9*speed km/h
    pid_agent = PID()
    action_times = 10
    ##########

    print("START!!")
        
    # steering.kinematics.publisher_(steering.kinematics.module_control_pub, "none", latch=True)
    # sleep(0.1)
    
    # Prepare a curses window control
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    
    start = time()
    count = 0
    while not rospy.is_shutdown(): 
        if count < 2.4:
            move_angle = pid_agent.controller(0)
        else:
            move_angle = pid_agent.controller(pid_agent.roll_state)
        
        ## Test ##
        sleep(1/action_times)
        env.step(np.degrees(move_angle))
        sleep(1/action_times)
        count = time() - start
        
        stdscr.clear()

        stdscr.addstr(f"Gogoro State (right side is positive value) \n")
        stdscr.addstr(f"Timer : {count:.2f} \n")

        stdscr.addstr(f"Speed       : {-pid_agent.speed_x:.2f} m/s \n")
        stdscr.addstr(f"Speed       : {-pid_agent.speed_x*3.6:.2f} km/h \n")
        stdscr.addstr(f"Yaw_Speed   : {-pid_agent.angvel_z:.2f} rad/s \n\n")

        stdscr.addstr(f"Roll  : {np.degrees(pid_agent.roll_state):.2f} deg\n")
        stdscr.addstr(f"Yaw   : {np.degrees(pid_agent.yaw_state):.2f} deg\n")
        stdscr.addstr(f"Steer : {pid_agent.action_angle:.2f} deg\n")

        
        c = stdscr.getch()
        
        if pid_agent.break_signal or c == ord('q'):
            break
    
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    
    print("Finished.")