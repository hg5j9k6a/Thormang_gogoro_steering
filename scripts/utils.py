import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_euler(x, y, z, w):
    t0      = +2.0 * (w * x + y * z)
    t1      = +1.0 - 2.0 * (x * x + y * y)
    roll    = math.atan2(t0, t1)
    t2      = +2.0 * (w * y - z * x)
    t2      = +1.0 if t2 > +1.0 else t2
    t2      = -1.0 if t2 < -1.0 else t2
    pitch   = math.asin(t2)
    t3      = +2.0 * (w * z + x * y)
    t4      = +1.0 - 2.0 * (y * y + z * z)
    yaw     = math.atan2(t3, t4)
    yaw     = yaw - math.pi
    if yaw < -math.pi:
        yaw = yaw + 2*math.pi
    return [roll, pitch, yaw]

def radToKmh(w_radius, rads):
    return 3.6 * w_radius * rads    

def rotateZ( theta ):
    rz = np.array( [ [ math.cos( theta ), - math.sin( theta ), 0, 0 ],
                           [ math.sin( theta ), math.cos( theta ), 0, 0 ],
                           [ 0, 0, 1, 0 ],
                           [ 0, 0, 0, 1 ] ] )
    return rz

def rotateY( theta ):
    ry = np.array( [ [ math.cos( theta ), 0, math.sin( theta ), 0 ],
                           [ 0, 1, 0, 0 ],
                           [ - math.sin( theta ), 0, math.cos( theta ), 0 ],
                           [ 0, 0, 0, 1 ] ] )
    return ry

def rotateX( theta ):
    rx = np.array( [ [ 1, 0, 0, 0 ],
                           [ 0, math.cos( theta ), - math.sin( theta ), 0 ],
                           [ 0, math.sin( theta ), math.cos( theta ), 0 ],
                           [ 0, 0, 0, 1 ] ] )
    return rx

def translate( dx, dy, dz ):
    t = np.array( [ [ 1, 0, 0, dx ],
                          [ 0, 1, 0, dy ],
                          [ 0, 0, 1, dz ],
                          [ 0, 0, 0, 1 ] ] )
    return t

def DHH( theta, d, a, alpha ):
    return rotateZ( theta ).dot( translate( 0, 0, d) ).dot( translate( a, 0, 0 ) ).dot( rotateX( alpha ) )

def plotCoordinateSystem( ax, length = 1.0, width = 1.0, A = None ):
    if ( A is None ):
        A = np.eye( 4 )
    xAxis = np.array( [ [ 0, 0, 0, 1], [ length, 0, 0, 1 ] ] ).T
    yAxis = np.array( [ [ 0, 0, 0, 1], [ 0, length, 0, 1 ] ] ).T
    zAxis = np.array( [ [ 0, 0, 0, 1], [ 0, 0, length, 1 ] ] ).T
    
    ax.plot( A.dot( xAxis )[0,:], A.dot( xAxis )[1,:], A.dot( xAxis )[2,:], 'r-', linewidth=width )
    ax.plot( A.dot( yAxis )[0,:], A.dot( yAxis )[1,:], A.dot( yAxis )[2,:], 'g-', linewidth=width )
    ax.plot( A.dot( zAxis )[0,:], A.dot( zAxis )[1,:], A.dot( zAxis )[2,:], 'b-', linewidth=width )

def drawLink(ax, A1, A2, width=1 ):
    x1 = A1.dot( np.array( [ 0, 0, 0, 1 ] ).T )
    x2 = A2.dot( np.array( [ 0, 0, 0, 1 ] ).T )
    c = np.vstack( (x1, x2 ) ).T
    # print('x1', x1, 'x2', x2, 'c', c)
    ax.plot( c[0,:], c[1,:], c[2,:], color='#50303030', linewidth=width )
    
def rxyz(A):
    if A[2,0] != 1 and A[2,0] != -1: 
        pitch_1 = -1*math.asin(A[2,0])
        pitch_2 = math.pi - pitch_1 
        roll_1 = math.atan2( A[2,1] / math.cos(pitch_1) , A[2,2] /math.cos(pitch_1) ) 
        roll_2 = math.atan2( A[2,1] / math.cos(pitch_2) , A[2,2] /math.cos(pitch_2) ) 
        yaw_1 = math.atan2( A[1,0] / math.cos(pitch_1) , A[0,0] / math.cos(pitch_1) )
        yaw_2 = math.atan2( A[1,0] / math.cos(pitch_2) , A[0,0] / math.cos(pitch_2) ) 

        # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
        # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info). 
        pitch = pitch_1 
        roll = roll_1
        yaw = yaw_1 
    else: 
        yaw = 0 # anything (we default this to zero)
        if A[2,0] == -1: 
            pitch = math.pi/2 
            roll = yaw + math.atan2(A[0,1],A[0,2]) 
        else: 
            pitch = -math.pi/2 
            roll = -1*yaw + math.atan2(-1*A[0,1],-1*A[0,2]) 

    # convert from radians to degrees
    # roll = roll*180/pi 
    # pitch = pitch*180/pi
    # yaw = yaw*180/pi 
    
    rxyz = [roll , pitch , yaw] 
    return rxyz