#https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
#https://gymnasium.farama.org/introduction/create_custom_env/
import gymnasium as gym
import numpy as np
from controller import Supervisor, Motor, GPS, Lidar, Camera, Accelerometer, Gyro
from typing import Optional
import math

class GridWorldEnv(gym.Env):

    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())


        self.f_left_motor = self.robot.getDevice('front left wheel') 
        self.f_right_motor = self.robot.getDevice('front right wheel')
        self.b_left_motor = self.robot.getDevice('back left wheel')
        self.b_right_motor = self.robot.getDevice('back right wheel')

        self.f_left_motor.setPosition(float('inf'))
        self.f_right_motor.setPosition(float('inf'))
        self.b_left_motor.setPosition(float('inf'))
        self.b_right_motor.setPosition(float('inf'))

        self.f_left_motor.setVelocity(0.0)
        self.f_right_motor.setVelocity(0.0)
        self.b_left_motor.setVelocity(0.0)
        self.b_right_motor.setVelocity(0.0)


        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)

        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)

        self.accelerometer = self.robot.getDevice('accelerometer')
        if self.accelerometer:
            self.accelerometer.enable(self.timestep)

        self.gyro = self.robot.getDevice('gyro')
        if self.gyro:
            self.gyro.enable(self.timestep)

        #Until this point, the code was taken from the controller_movement init
        #Now let's get this shit done

        self.robot_node = self.robot.getFromDef("ROBOT")
        self.initial_position = self.robot_node.getPosition()
        self.initial_rotation = self.robot_node.getOrientation()

        self.target_node = self.robot.getFromDef("TARGET")
        self.target_initial_position = self.target_node.getPosition()

        self.observation_space = gym.spaces.Dict(
            {
                "lidar": gym.spaces.Box(low=0.0, high=np.inf, shape=(10,), dtype=np.float32),
                "goal_info": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            }
        )
        self.action_space = gym.spaces.Box(low=-6.4, high=6.4, shape=(2,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed) # IMPORTANT: Must call this first to seed the random number generator

        trans_field = self.robot_node.getField("translation")
        rot_field = self.robot_node.getField("rotation")
        trans_field.setSFVec3f(self.initial_position)

        self.f_left_motor.setVelocity(0.0)
        self.f_right_motor.setVelocity(0.0)
        self.b_left_motor.setVelocity(0.0)
        self.b_right_motor.setVelocity(0.0)

        self.robot.step(self.timestep)

        observation = self._get_obs()
        #add some goal info (info = ...)
        

        return observation

    def _get_obs(self):
        pos = self.gps.getValues()
        robot_x, robot_z = pos[0], pos[2]
        target_x, target_z = self.target_pos[0], self.target_pos[2]

        #calculate distance to target (Euclidian distance)
        dist = math.sqrt((target_x - robot_x)**2 + (target_z - robot_z)**2)
        angle_to_target = math.atan2(target_z - robot_z, target_x - robot_x)

        #LIDAR
        lidar_data = [1.0, 1.0, 1.0] #Lidar simplified data to left, center, right
        if self.lidar:
            raw_lidar_data = self.lidar.getRangeImage()
            if raw_lidar_data:
                n = len(raw_lidar_data)

                left = raw_lidar_data[:n//3]
                center = raw_lidar_data[n//3:2*n//3]
                right = raw_lidar_data[2*n//3:]

                lidar_data = [min(left), min(center), min(right)]
        
        observation = {
            "lidar": np.array(lidar_data, dtype=np.float32),
            "goal_info": np.array([dist, angle_to_target], dtype=np.float32)
        }

        return observation