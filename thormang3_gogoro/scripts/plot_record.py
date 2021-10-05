import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
import numpy as np
import os
import rospkg

rospack = rospkg.RosPack()
PKG_PATH = rospack.get_path('thormang3_gogoro') + "data/0706_PID_12_0"
# Path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/data/0706_PID_12_0"
file_name = "control_record.npy"

# Path_con = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
# right_non_position = np.load(os.path.join(Path_con,"right_non_gasing_position.npy"))
# print(len(right_non_position))
# print(right_non_position)

fig = plt.figure( figsize=(10,15) )
# ax = fig.add_subplot(111)
ax2 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)

record_data = np.load(os.path.join(Path,file_name))
xs = np.arange(0,len(record_data))
steering = record_data[:,0]
error = record_data[:,1]

del_steering = []

for i in range(1,len(steering)):
    del_steering.append(steering[i]-steering[i-1])
del_steering = np.array(del_steering)
del_xs = np.arange(0,len(del_steering))
print("Max del_deg:",np.max(del_steering))
print("Min del_deg:",np.min(del_steering))

# print(error)
# x_major_locator=MultipleLocator(1)
# ax.xaxis.set_major_locator(x_major_locator)
# ax2.xaxis.set_major_locator(x_major_locator)
# ax3.xaxis.set_major_locator(x_major_locator)

# ax.plot(xs[:],-np.radians((error[:])),label='Tilt_Error',color="g")
ax2.plot(xs[:],-np.radians((steering[:])),label='Steering_angle',color="b")
ax3.plot(del_xs[:],-np.radians((del_steering[:])),label='Steering Angle Velocity',color="r")

# ax.set_title("Φ-Tilt",fontsize=12)
ax2.set_title("δ-Steering Angle",fontsize=12)
ax3.set_title("Steering Angle Velocity",fontsize=12)

# ax.set_xlabel('Time(Iteration)', fontsize=8)
# ax.set_ylabel('Radians', fontsize=8)

ax2.set_xlabel('Time(Iteration)', fontsize=8)
ax2.set_ylabel('Radians', fontsize=8)

ax3.set_xlabel('Time(Iteration)', fontsize=8)
ax3.set_ylabel('Radians', fontsize=8)

# ax.legend()
ax2.legend()
ax3.legend()

plt.show()