#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64,String,Bool
from pioneer_kinematics.kinematics import Kinematics
from sensor_msgs.msg import JointState

from time import sleep
import numpy as np
import socket
import sys #for exit
import os

break_pub = rospy.Publisher("/thormang3_gogoro/steering/break", Bool, queue_size=2)
start_pub = rospy.Publisher("/thormang3_gogoro/steering/start", Bool, queue_size=2)

#create an AF_INET, STREAM socket (TCP)
try:
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,socket.IPPROTO_UDP)
except socket.error as msg:
    print('Failed to create socket. Error code: ', str(msg[0]) ,' , Error message : ', msg[1])

print('Socket Created')

client.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
client.bind(("",8182))
print("Created client complete")

_node_name = 'emergency_call'
rospy.init_node(_node_name, anonymous=True)

# position_path = "/home/robotis/mixed_ws/src/gogoro_emergency/config"
# right_break_position  = np.load(os.path.join(position_path,"right_break.npy"))
# left_break_position   = np.load(os.path.join(position_path,"left_break.npy"))

# kinematics = Kinematics()
# arm_joint = ["l_arm_el_y","l_arm_grip","l_arm_sh_p1","l_arm_sh_p2","l_arm_sh_r","l_arm_wr_p","l_arm_wr_r","l_arm_wr_y",
#             "r_arm_el_y","r_arm_grip","r_arm_sh_p1","r_arm_sh_p2","r_arm_sh_r","r_arm_wr_p","r_arm_wr_r","r_arm_wr_y"]
# right_break = right_break_position
# left_break  = left_break_position

# arm_break   = np.hstack([left_break,right_break])

# joint           =   JointState()
# joint.name      =   arm_joint

# joint.position  =   arm_break

# joint.velocity  =   [ 0 for _ in arm_joint ]
# joint.effort    =   [ 0 for _ in arm_joint ]

while not rospy.is_shutdown():
# while True:
    data,addr = client.recvfrom(1024)
    # print("received message:",data.decode())
    state = data.decode().split(" ")[-1].split("\n")[0]
    print(state)
    
    if state == "DANGER":
        break_pub.publish(True)
        
    #     kinematics.publisher_(kinematics.module_control_pub, "none", latch=True) 
    #     kinematics.publisher_(kinematics.set_joint_pub, joint, latch=False)
        
    else:
        break_pub.publish(False)
        
    if state == "Run":
        start_pub.publish(True)
    else:
        start_pub.publish(False)
    
