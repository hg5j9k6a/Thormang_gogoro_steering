#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64,Float32MultiArray,Bool
import curses
from time import sleep
import numpy as np

class Data_transmission():
    def __init__(self):
        self.speed          = 0
        self.yaw_speed      = 0
        self.steering_angle = 0
        self.roll_state     = 0
        self.yaw_state      = 0
        
        self.break_bool = False
        
        self.break_pub = rospy.Publisher("/thormang3_gogoro/steering/break", Bool, queue_size=2)
        rospy.Subscriber('/thormang3_gogoro/steering/data', Float32MultiArray, self._data_receiver)
        
    def _data_receiver(self,msg):
        self.speed          = msg.data[0]
        self.yaw_speed      = msg.data[1]
        self.roll_state     = msg.data[2]
        self.yaw_state      = msg.data[3]
        self.steering_angle = msg.data[4]
        
        print(f"Gogoro State (right side is positive value) ")
        print("")
        print(f"Speed       : {-data_tran.speed:.2f} m/s ")
        print(f"Speed       : {-data_tran.speed*3.6:.2f} km/h ")
        print(f"Yaw_Speed   : {-data_tran.yaw_speed:.2f} rad/s ")
        print("")
        print(f"Roll  : {data_tran.roll_state:.2f} deg")
        print(f"Yaw   : {data_tran.yaw_state:.2f} deg")
        print(f"Steer : {data_tran.steering_angle:.2f} deg")
        print("--------------------\n")        


    def break_signal(self):
        for i in range(3):
            self.break_pub.publish(True)
            sleep(0.1)
        
if __name__ == "__main__":
    _node_name = 'Data_Visual'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))
    
    data_tran = Data_transmission()
    
    rate = rospy.Rate(30)
    
    # Prepare a curses window control
    
    # print("Finished.")
    rospy.spin()
    
    