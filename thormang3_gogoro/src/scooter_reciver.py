#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64,String,Bool

import socket
import sys #for exit

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

_node_name = 'Cellphone_State'
rospy.init_node(_node_name, anonymous=True)

while not rospy.is_shutdown():
    data,addr = client.recvfrom(1024)
    # print("received message:",data.decode())
    state = data.decode().split(" ")[-1].split("\n")[0]
    print(state)
    
    if state == "DANGER":
        break_pub.publish(True)
    else:
        break_pub.publish(False)
        
    if state == "Run":
        start_pub.publish(True)
    else:
        start_pub.publish(False)