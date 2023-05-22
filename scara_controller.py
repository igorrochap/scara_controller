"""scara_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import numpy as np
from controller import Supervisor  # type: ignore

np.set_printoptions(precision=4, suppress=True)


def c(x):
    return np.cos(x)


def s(x):
    return np.sin(x)


def fkine(q):
    """
    # TODO
    Implement the forward kinematics of the SCARA robot.
    Input: q - joint angles (list of 4 floats)
    Output: T - transformation matrix (4x4 numpy array)
    """
    [q_1, q_2, q_3, q_4] = q
    
    d_0 = 0.65
    d_1 = 0.10
    d_2 = 0.50
    d_3 = 0.50
    d_4 = -0.22
    
    T_0 = np.array([
               [1, 0, 0, 0], 
               [0, 1, 0 ,0],
               [0, 0, 1, d0],
               [0, 0, 0, 1]
               ])
    
    T_1 = np.array([
               [np.cos(q1), -np.sin(q1), 0, d2], 
               [np.sin(q1), np.cos(q1), 0, 0],
               [0, 0, 1, d1],
               [0, 0, 0, 1]
               ])
      
      
    T_2 = np.array([
               [np.cos(q2), -np.sin(q2), 0, d3], 
               [np.sin(q2), np.cos(q2), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]
               ]) 
               
    T_3 = np.array([
               [1, 0, 0, 0], 
               [0, np.cos(q3), -np.sin(q3), 0],
               [0, np.sin(q3), np.cos(q3), -(d4 + q2)],
               [0, 0, 0, 1]
               ]) 
               
               
    return np.dot(np.dot(np.dot(T_0,T_1), T_2), T_3)           
               
               
                         

def invkine(x, y, z, phi):
    """
    # TODO
    Implement the inverse kinematics of the SCARA robot.
    Input: x, y, z, phi - desired end-effector pose
    Output: q - joint angles (list of 4 floats)
    """    
    a_1 = 0.50
    a_2 = 0.50 
    d_0 = 0.65
    d_1 = 0.10
    d_4 = 0.22
    
    c = (x**2 + y**2 - a_1**2 - a_2**2) / (2 * a_1 * a_2)
    
    if c < -1 or c > 1:
        raise ValueError("Out of range!")
        
       
    q_2 = np.arctan2(np.sqrt(1 - c**2), c)
    k_1 = a_1 + a_2 * np.cos(q_2)
    k_2 = a_2 * np.sin(q_2)
    q_1 = np.arctan2(y, x) - np.arctan2(k_2, k_1)
    q_3 = d_0 + d_1 - d_4 - z
    q_4 = phi - q_1 - q_2
    
    return [q_1, q_2, q_3, q_4]


class Scara:
    def __init__(self):
        # create the Robot instance.
        self.robot = Supervisor()

        # get the time step of the current world.
        self.timestep = int(self.robot.getBasicTimeStep())

        # get joint motor devices
        self.joints = [self.robot.getDevice("joint%d" % i) for i in range(1, 5)]

        # get duck reference
        self.duck = self.robot.getFromDef("DUCK")

        # get gripper reference
        self.gripper = self.robot.getFromDef("GRIPPER")

        self.grasp = False
        self.grasp_prev = False

    def set_position(self, q):
        """
        Set the joint positions of the SCARA robot.
        Input: q - joint angles (list of 4 floats)
        """
        for joint, value in zip(self.joints, q):
            joint.setPosition(value)

    def is_colliding(self, ds=0.15):
        """
        Check if the gripper is colliding with the duck.
        Input: ds - safety distance (float)
        Output: new_pos - new gripper position (list of 3 floats)
                new_yaw - new gripper yaw (float)
                colliding - True if colliding, False otherwise (bool)
        """
        dp = np.array(self.duck.getPose()).reshape(4, 4)[:-1, -1]
        gt = np.array(self.gripper.getPose()).reshape(4, 4)
        gp = gt[:-1, -1]
        gy = np.arctan2(gt[1, 0], gt[0, 0])
        return (
            (gp + np.array([0.0, 0.0, -0.5 * ds])).tolist(),
            gy,
            (np.linalg.norm(dp - gp) < ds),
        )

    def step(self):
        """
        Perform one simulation step.
        Output: -1 if Webots is stopping the controller, 0 otherwise (int)
        """
        if self.grasp is True and self.grasp_prev is False:
            print("GRASP STARTED")
            self.grasp_prev = True
        elif self.grasp is False and self.grasp_prev is True:
            print("GRASP ENDED")
            self.grasp_prev = False

        if self.grasp:
            new_pos, new_yaw, colliding = self.is_colliding()
            self.duck.resetPhysics()
            if colliding:
                self.duck.getField("translation").setSFVec3f(new_pos)
                self.duck.getField("rotation").setSFRotation([0.0, 0.0, 1.0, new_yaw])

        return self.robot.step(self.timestep)

    def delay(self, ms):
        """
        Delay the simulation for a given time.
        Input: ms - delay time in milliseconds (int)
        """
        counter = ms / self.timestep
        while (counter > 0) and (self.step() != -1):
            counter -= 1

    def hold(self):
        """
        Hold the duck.
        """
        self.grasp = True

    def release(self):
        """
        Release the duck.
        """
        self.grasp = False

    def getDuckPose(self):
        """
        Get the duck pose.
        Output: position - duck position (list of 3 floats)
                yaw - duck yaw (float)
        """
        pose = np.array(self.duck.getPose()).reshape(4, 4)
        position = pose[:-1, -1].tolist()
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        return position + [yaw]


if __name__ == "__main__":

    def hold_duck(scara):
        duck_position = scara.getDuckPose()                  
        angles = invkine(*duck_position)        
        scara.set_position(angles)
        scara.delay(4500)
        scara.hold()
        
    def release_duck(scara):
        box_position = [0.85, -0.3, 0.4]
        height_diff = 0.3 
        box_position = [box_position[0], box_position[1], box_position[2] + height_diff, 0]
        angles = invkine(*box_position)
        scara.set_position(angles)
        scara.delay(4500)
        scara.release()
         
                           
    scara = Scara()
    # Main loop:
    # Perform simulation steps until Webots is stopping the controller
    while scara.step() != -1:
        # TODO
        # Implement your code here
       hold_duck(scara)
       release_duck(scara)
                
       break

    # Enter here exit cleanup code.

