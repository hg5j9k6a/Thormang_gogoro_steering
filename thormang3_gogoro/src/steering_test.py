#! /usr/bin/env python3

import rospy
from pioneer_kinematics.kinematics import Kinematics
from thormang3_manipulation_module_msgs.msg import KinematicsPose
from robotis_controller_msgs.msg import JointCtrlModule
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

from time import sleep
import numpy as np
import sys

def countdownSleep(seconds):
    while seconds > 0:
        print("Will continue in {0}...".format(seconds))
        sleep(1)
        seconds -= 1

def log(tag, msg):
    rospy.loginfo(tag + ': ' + msg)

class Gripper:
    TAG = '[Gripper]'

    MIN_LIMIT = 0.0
    MAX_LIMIT = 1.1
    
    def __init__(self):
        self.left_name = 'l_arm_grip'
        self.right_name = 'r_arm_grip'

        # Joint state publisher
        self.jointStatesPublisher = rospy.Publisher('/robotis/set_joint_states', 
                JointState, queue_size=1)
        self.jointModulePublisher = rospy.Publisher('/robotis/set_joint_ctrl_modules', 
                JointCtrlModule, queue_size=1)

        # Set grippers control module to none
        self._setGrippersNoneModule()

        # Also setup a subscriber and callback for easy debugging
        rospy.Subscriber('/thormang3_gogoro/grippers', Float64, 
                self._grippers_cb)

    def _grippers_cb(self, data):
        self.setGrippers(left=data.data, right=data.data)

    def _setGrippersNoneModule(self):
        msg = JointCtrlModule()
        msg.joint_name = [self.left_name, self.right_name]
        msg.module_name = ['none', 'none']

        self.jointModulePublisher.publish(msg)

    def setGrippers(self, left, right):
        ''' Set grippers as a percentage of maximum and minimum opening

        '''
        if left < 0.0 or left > 1.0 or \
           right < 0.0 or right > 1.0:
            log(self.TAG, "value for `setGrippers` outside of range. [0-1]")

        # Convert percentage to range, take the easy way since `MIN_LIMIT` is 0.0
        right_val = right * self.MAX_LIMIT
        left_val = left * self.MAX_LIMIT
        
        jointStateMsg = JointState()
        jointStateMsg.name = [self.left_name, self.right_name]
        jointStateMsg.position = [left_val, right_val]

        self.jointStatesPublisher.publish(jointStateMsg)

class Thormang3Steering:
    TAG = '[Thormang3Steering]' 
    def __init__(self):
        ### Parameters
        # X (forward) front of robot is positive
        # Y (sideways) left of robot is positive
        # Z (up-down) up of robot is positive
        self.mid_point = np.array([0.3, 0.0, 1.0]) # in x, y, z
        self.y_spacing = 0.35 # Spacing between the arms
        self.rotation_angle = 0.0 # Around mid_point, In degrees
        self.right_pitch = 0.0 # In degrees
        self.roll = 90 # In degrees (for both arms)

        self.offsetMatchBars = 10.0 # In degrees

        self.time = 3.0
        ###
        # Kinematics control object
        self.kinematics = Kinematics()
        self.kinematics.publisher_(self.kinematics.module_control_pub, "manipulation_module", latch=True)  

        # Send `ini_pose`
        log(self.TAG, "Moving to ini pose...")
        self.kinematics.publisher_(self.kinematics.send_ini_pose_msg_pub, "ini_pose", latch=True)
        #sleep(3)

        rospy.Subscriber('/thormang3_gogoro/steering/y_spacing', 
                Float64, self._y_spacing_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/mid_point', 
                Vector3, self._mid_point_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/steering_angle', 
                Float64, self._rotation_angle_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/right_pitch', 
                Float64, self._right_gripper_pitch_cb)

    def _y_spacing_cb(self, data):
        log(self.TAG, "new `y_spacing`: {0}".format(data.data))
        self.y_spacing = data.data 
        self.move()

    def _mid_point_cb(self, data):
        log(self.TAG, "new `mid_point`: {0}".format((data.x, data.y, data.z)))
        self.mid_point[0] = data.x
        self.mid_point[1] = data.y
        self.mid_point[2] = data.z

        self.move()

    def _rotation_angle_cb(self, data):
        log(self.TAG, "new `rotation_angle`: {0}".format(data.data))
        self.rotation_angle = data.data
        self.move(time=1.0)

    def _right_gripper_pitch_cb(self, data):
        log(self.TAG, "new `right_pitch`: {0}".format(data.data))
        self.right_pitch = data.data
        self.move()

    def setMidPoint(self, mid_point):
        self.mid_point = mid_point

    def setYSpacing(self, y_spacing):
        self.y_spacing = y_spacing

    def setRotation(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def changeParams(self, mid_point, y_spacing, rotation_angle):
        self.mid_point = mid_point
        self.y_spacing = y_spacing
        self.rotation_angle = rotation_angle

    def setRightGripperPitch(self, pitch):
        self.right_pitch = pitch

    def move(self, time=None):
        ''' Using `mid_point` and `y_spacing` will compute the left and right
        arm positions and send the commands

        '''

        if time != None:
            time = time
        else:
            time = self.time

        # Compute new position based on parameters
        h = self.y_spacing / 2.
        x_inc = np.sin(np.radians(self.rotation_angle)) * h
        y_inc = np.cos(np.radians(self.rotation_angle)) * h

        l_x = self.mid_point[0] - x_inc
        l_y = self.mid_point[1] + y_inc
        l_z = self.mid_point[2]
        r_x = self.mid_point[0] + x_inc
        r_y = self.mid_point[1] - y_inc
        r_z = self.mid_point[2]

        # Send new positions
        self.kinematics.set_kinematics_pose("left_arm", time,
                **{'x': l_x, 'y': l_y, 'z': l_z,
                   'roll': self.roll, 'pitch': 0.0, 'yaw': self.rotation_angle + self.offsetMatchBars})
        # Note the mirrored signal for the roll in the right arm
        self.kinematics.set_kinematics_pose("right_arm", time,
                **{'x': r_x, 'y': r_y, 'z': r_z,
                   'roll': -self.roll, 'pitch': self.right_pitch, 'yaw': self.rotation_angle - self.offsetMatchBars})

    def resetPos(self):
        log(self.TAG, "resetPos")
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': 0.50, 'y':  0.60, 'z': 1.00, 'roll': 90.00, 'pitch': 0.00, 'yaw': 0.00 })

        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': 0.50, 'y':  -0.60, 'z': 1.00, 'roll': -90.00, 'pitch': 0.00, 'yaw': 0.00 })
        sleep(4)

if __name__ == "__main__":
    _node_name = 'kinematics_test'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))

    # TODO: These are not definitive positions!
    setup_pos1 = np.array([0.3, 0.0, 0.850])
    grab_handle_pos = np.array([0.47, 0.0, 0.850])

    gripper = Gripper()

    #gripper.setGrippers(left=0.0, right=0.0)
    #rospy.spin()
    #sys.exit(1)

    steering = Thormang3Steering()
    steering.resetPos()

    # Spacing is fixed, for now
    steering.setYSpacing(0.5)
    steering.setRightGripperPitch(25)
    # Grippers start open
    gripper.setGrippers(left=0.2, right=0.2)

    input("Enter to next....")
    steering.setMidPoint(setup_pos1)
    steering.move()
    sleep(4.0)

    input("Enter to move to handle pos....")
    steering.setMidPoint(grab_handle_pos)
    steering.move()
    sleep(4.0)
    
    input("Enter to grab handle....")
    gripper.setGrippers(left=0.95, right=0.95)

    input("Rotate to right side...")
    val = 2
    for _ in range(2):
        steering.setRotation(15)
        steering.move(time=val)
        sleep(val + 2.0)

        steering.setRotation(-15)
        steering.move(time=val)
        sleep(val + 2.0)

        steering.setRotation(0.0)
        steering.move(time=val)
        sleep(val + 2.0)
        val -= 1

    gripper.setGrippers(left=0.2, right=0.2)
    steering.setMidPoint(setup_pos1)
    steering.move()
    sleep(4.0)

    rospy.spin()
