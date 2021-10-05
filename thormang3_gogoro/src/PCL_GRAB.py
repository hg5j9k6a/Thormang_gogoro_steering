#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pcl
# import pptk
from sklearn.cluster import KMeans


def grab_position(data_pcl):
    
    data_z   = data_pcl[:,2]
    z_min = 0.0 #np.round( np.min(data_z), 1) + 0.08 # 0.08
    data_pcl = data_pcl[np.where( (data_z >= z_min) )]
    
    data_x   = data_pcl[:,0]
    x_min = 0.3 #np.round( np.min(data_x), 1) + 0.1 # 0.5
    x_max = 0.5 #np.round( np.max(data_x), 1) - 0.3 # 0.3
    data_pcl = data_pcl[np.where( (data_x >= x_min) & (data_x <= x_max))]
    
    data_y   = data_pcl[:,1]
    y_min = -0.45 #np.round( np.min(data_y), 1) + 0.1 # 0.2
    y_max =  0.45 #np.round( np.max(data_y), 1) - 0.1 # 1.2 
    data_pcl = data_pcl[np.where( (data_y >= y_min) & (data_y <= y_max))]
    
    
    data_y_left = data_pcl[:,1]
    y_left = -0.2
    data_y_left = data_pcl[np.where( (data_y_left <= y_left) )]

    mean_left_x = np.mean(data_y_left[:,0])    
    mean_left_y = np.mean(data_y_left[:,1])
    
    data_y_right = data_pcl[:,1]
    y_right = 0.2
    data_y_right = data_pcl[np.where( (data_y_right >= y_right) )]
    
    mean_right_x = np.mean(data_y_right[:,0])    
    mean_right_y = np.mean(data_y_right[:,1])
    
    # print(data_y_left)
    # print(data_y_right)
    
    data_pcl = []
    
    for i in range(len(data_y_left)):
        data_pcl.append(data_y_left[i])
    for i in range(len(data_y_right)):
        data_pcl.append(data_y_right[i])
    
    data_pcl = np.array(data_pcl)
    # cloud = pcl.PointCloud(np.array(data_pcl[:,:3], dtype=np.float32))
    # cloud_filtered = cloud
    
    mean_z = np.mean(data_pcl[:,2])
    
    
    print("mean_left_x	:",mean_left_x)
    print("mean_left_y	:",mean_left_y)
    print("mean_right_x	:",mean_right_x)
    print("mean_right_y	:",mean_right_y)
    print("mean_z		:",mean_z)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_pcl[:,0],data_pcl[:,1],data_pcl[:,2],s=1)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
 
    plt.show()
    
    # left_arm = {'x': mean_left_x, 'y':  mean_left_y, 'z': 0.82+mean_z, 'roll': -90.00, 'pitch': 0.00, 'yaw': 0.00}
    # right_arm = {'x': mean_right_x, 'y':  mean_right_y, 'z': 0.82+mean_z, 'roll': 90.00, 'pitch': 0.00, 'yaw': 0.00}
    
    left_arm = {'x': 0.4, 'y':  mean_left_y, 'z': 0.82+mean_z, 'roll': -90.00, 'pitch': 0.00, 'yaw': 0.00}
    right_arm = {'x': 0.4, 'y':  mean_right_y, 'z': 0.82+mean_z, 'roll': 90.00, 'pitch': 0.00, 'yaw': 0.00}
    
    return left_arm , right_arm
