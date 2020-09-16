#-*-coding:utf-8-*-

"""
Program to autonomusly collect dataset using Pure Pursuit
"""

# Import Necessary Modules
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import vrep
import sys

#Parameters tuned for pure pursuit
k = 0.08# look forward gain
Lfc =0.2 # look-ahead distance
Kp = 1.0  # speed propotional gain
dt = 0.25 # [s]
L = 0.07772

# just in case, close all opened connections
vrep.simxFinish(-1) 
# start a connection
clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5) 
if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

#Creates the handle for each object
err_code,naked_steering_car = vrep.simxGetObjectHandle(clientID,'nakedAckermannSteeringCar', vrep.simx_opmode_blocking)
err_code,r_motor_handle = vrep.simxGetObjectHandle(clientID,'nakedCar_freeAxisRight', vrep.simx_opmode_blocking)
err_code,l_motor_handle = vrep.simxGetObjectHandle(clientID,'nakedCar_freeAxisLeft', vrep.simx_opmode_blocking)
err_code,l_motor_steering = vrep.simxGetObjectHandle(clientID,'nakedCar_steeringLeft', vrep.simx_opmode_blocking)
err_code,r_motor_steering = vrep.simxGetObjectHandle(clientID,'nakedCar_steeringRight', vrep.simx_opmode_blocking)
err_code,dummy = vrep.simxGetObjectHandle(clientID,'Dummy', vrep.simx_opmode_blocking)
err_code,dummy0 = vrep.simxGetObjectHandle(clientID,'Dummy0', vrep.simx_opmode_blocking)
err_code,dummy1 = vrep.simxGetObjectHandle(clientID,'Dummy1', vrep.simx_opmode_blocking)
err_code,fl_motor_handle = vrep.simxGetObjectHandle(clientID,'nakedCar_motorLeft', vrep.simx_opmode_blocking)
err_code,fr_motor_handle = vrep.simxGetObjectHandle(clientID,'nakedCar_motorRight', vrep.simx_opmode_blocking)
err_code,cam_handle=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_blocking)
err,res,image=vrep.simxGetVisionSensorImage(clientID,cam_handle,0,vrep.simx_opmode_streaming)

while(True):
    err,res,image=vrep.simxGetVisionSensorImage(clientID,cam_handle,0,vrep.simx_opmode_buffer)
    if err==vrep.simx_return_ok:
        break

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

#Function to update the pose of the vehicle
def update(state,delta,a):

    RobotPos=vrep.simxGetObjectPosition(clientID, dummy, -1, vrep.simx_opmode_blocking) 
    Orientation=vrep.simxGetObjectOrientation(clientID,dummy, -1, vrep.simx_opmode_blocking)
    yaw=Orientation[1][1]
    state.x=RobotPos[1][0]
    state.y=RobotPos[1][1]
    state.yaw =yaw
    state.v=vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,1,vrep.simx_opmode_blocking)
    return state
    
def PIDControl(target, current):
    a = Kp * (target - current)
    return a


def pure_pursuit_control(state, cx, cy, pind):

    ind = calc_target_index(state, cx, cy)

    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    if state.v < 0:  # back
        alpha = math.pi - alpha

    Lf = k * state.v + Lfc
    #Calculates the steering angle
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)
    return delta, ind


def calc_target_index(state, cx, cy):

    # search nearest point index
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0
    Lf = k * state.v + Lfc
    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind] - state.x
        dy = cy[ind] - state.y
        L = math.sqrt(dx ** 2 + dy ** 2)
        ind += 1
    return ind

#Function to count the number of files in a folder
def Count(Folder):
    Command = os.listdir(Folder)
    x = len(Command)
    return x

#Function to collect the images
def create_dataset(value):
    err,res,image=vrep.simxGetVisionSensorImage(clientID,cam_handle,0,vrep.simx_opmode_oneshot)
    img=np.array(image,dtype=np.uint8)
    img.resize([res[0],res[1],3])
    #Specify the path to save the file
    path='Path to save '+str(value)
    a = Count(path)
    cv2.imwrite(os.path.join(path,'Image%05d.jpeg'%(int(a))),img)
    
def SteeringAngle(di):
    Angle=90+di
    if Angle >= 0 and Angle < 89.9:
        Value = 5
    elif Angle >= 89.9 and Angle < 90.1 :
        Value = 4
    elif Angle > 90.1 and Angle <=180:
        Value = 3
    else:
        Value=9
    create_dataset(Value)
  
def main():
    #Start the Simulation
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
    DummyPos1=vrep.simxGetObjectPosition(clientID,dummy1, -1, vrep.simx_opmode_blocking)
    DummyPos0=vrep.simxGetObjectPosition(clientID,dummy0, -1, vrep.simx_opmode_blocking)
    RobotPos=vrep.simxGetObjectPosition(clientID,dummy , -1, vrep.simx_opmode_blocking) 
    Orientation=vrep.simxGetObjectOrientation(clientID,dummy, dummy1, vrep.simx_opmode_blocking)
    yaw=Orientation[1][0]
    #Path points to follow
    cx =np.arange(DummyPos0[1][0],DummyPos1[1][0],0.01)#DummyPos0[1][0]
    cy =([DummyPos0[1][1]]*len(cx)*1)
    target_speed = 1  # [m/s]
    # initial state
    state = State(x=RobotPos[1][0], y= RobotPos[1][0],yaw=yaw, v=0)
    lastIndex = len(cx)
    tim = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)
    while lastIndex > target_ind :
        #Set the velocity of the wheels
        v=vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,1,vrep.simx_opmode_blocking)
        v=vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,1,vrep.simx_opmode_blocking)
        ai = PIDControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        SteeringAngle(di)
        #Sets the steering angle
        vrep.simxSetJointTargetPosition(clientID,r_motor_steering,di,vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(clientID,l_motor_steering,di,vrep.simx_opmode_blocking)
        #Vehicle State
        state.x=RobotPos[1][0]
        state.y=RobotPos[1][1]
        state.yaw=yaw
        state = update(state,di,ai)
        if state.x > DummyPos1[1][0]:
            print(state.x,DummyPos1[1][0])
            create_dataset(9)
            vrep.simxSetJointTargetPosition(clientID,r_motor_steering,0,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetPosition(clientID,l_motor_steering,0,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID,l_motor_handle,0,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID,r_motor_handle,0,vrep.simx_opmode_blocking)
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
            break
    print("Simulation Finished")

if __name__ == '__main__':
    print("Pure pursuit Path Tracking Simulation Start")
    main()

