import os
import sys

# --- CONFIGURACIÓN PARA LA CARPETA UNIFICADA DE WEBOTS ---
webots_home = os.environ.get("WEBOTS_HOME", "C:\\Program Files\\Webots")
webots_python_path = os.path.join(webots_home, "lib", "controller", "Python")

# Asegurarnos de que la ruta exista y añadirla a Python
if os.path.exists(webots_python_path) and webots_python_path not in sys.path:
    sys.path.append(webots_python_path)
# También aseguramos la ruta general de librerías de control
webots_controller_lib = os.path.join(webots_home, "lib", "controller")
if webots_controller_lib not in sys.path:
    sys.path.append(webots_controller_lib)
# ---------------------------------------------------------

#https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
#https://gymnasium.farama.org/introduction/create_custom_env/
import gymnasium as gym
import numpy as np
from controller import Supervisor, Motor, GPS, Lidar, Camera, Accelerometer, Gyro
from typing import Optional
import math

os.environ["WEBOTS_ROBOT_NAME"] = "supervisor_controller"

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
        rot_field = self.robot_node.getField("rotation")
        self.initial_rotation = rot_field.getSFRotation()

        self.target_node = self.robot.getFromDef("TARGET")
        self.target_initial_position = self.target_node.getPosition()

        self.observation_space = gym.spaces.Dict(
            {
                "lidar": gym.spaces.Box(low=0.0, high=5.0, shape=(3,), dtype=np.float32),
                "goal_info": gym.spaces.Box(low=-np.pi, high=np.inf, shape=(2,), dtype=np.float32)
            }
        )
        self.action_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32)
        self.prev_v_left = 0.0
        self.prev_v_right = 0.0
        self.previous_distance = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed) # IMPORTANT: Must call this first to seed the random number generator

        trans_field = self.robot_node.getField("translation")
        rot_field = self.robot_node.getField("rotation")
        trans_field.setSFVec3f(self.initial_position)
        rot_field.setSFRotation(self.initial_rotation)

        self.robot_node.resetPhysics()
        
        self.f_left_motor.setVelocity(0.0)
        self.f_right_motor.setVelocity(0.0)
        self.b_left_motor.setVelocity(0.0)
        self.b_right_motor.setVelocity(0.0)

        self.prev_v_left = 0.0
        self.prev_v_right = 0.0

        self.robot.step(self.timestep)

        observation = self._get_obs()
        self.previous_distance = observation["goal_info"][0]
        info = {}
        

        return observation, info

    def _get_obs(self):
        pos = self.gps.getValues()
        if pos is None or any(math.isnan(v) or math.isinf(v) for v in pos):
            robot_x, robot_y = 0.0, 0.0
        else:
            robot_x, robot_y = pos[0], pos[1]
        target_x, target_y = self.target_initial_position[0], self.target_initial_position[1]

        #calculate distance to target (Euclidian distance)
        dist = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
        if math.isnan(dist) or math.isinf(dist):
            dist = 0.0

        orientation = self.robot_node.getOrientation()
        if orientation is None or len(orientation) < 9 or any(math.isnan(v) or math.isinf(v) for v in orientation):
            robot_yaw = 0.0
        else:
            robot_yaw = math.atan2(orientation[3], orientation[0]) #In a Z-up (ENU) system, the frontal angle in the XY plane is obtained from [3] and [0]
        
        angle_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
        heading_error = angle_to_target - robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        if math.isnan(heading_error) or math.isinf(heading_error):
            heading_error = 0.0

        #LIDAR
        lidar_data = [5.0, 5.0, 5.0] #Lidar simplified data to left, center, right
        if self.lidar:
            raw_lidar_data = self.lidar.getRangeImage()
            if raw_lidar_data:
                cleaned_lidar = []
                for d in raw_lidar_data:
                    if math.isinf(d) or math.isnan(d):
                        cleaned_lidar.append(5.0)
                    else:
                        cleaned_lidar.append(float(np.clip(d, 0.0, 5.0)))

                if len(cleaned_lidar) > 0:
                    n = len(cleaned_lidar)
                    left = cleaned_lidar[:n//3]
                    center = cleaned_lidar[n//3:2*n//3]
                    right = cleaned_lidar[2*n//3:]
                    
                    lidar_data = [
                        min(left) if left else 5.0,
                        min(center) if center else 5.0,
                        min(right) if right else 5.0
                    ]
        
        observation = {
            "lidar": np.array(lidar_data, dtype=np.float32),
            "goal_info": np.array([dist, heading_error], dtype=np.float32)
        }

        return observation

    def step(self, action):
        target_v_left = action[0]
        target_v_right = action[1]

        # ACTION SMOOTHING (Inertia filter) – yep, Gemini idea
        alpha = 0.3
        v_left = self.prev_v_left * alpha + target_v_left * (1 - alpha)
        v_right = self.prev_v_right * alpha + target_v_right * (1 - alpha)

        self.prev_v_left = v_left
        self.prev_v_right = v_right
        
        self.f_left_motor.setVelocity(v_left)
        self.f_right_motor.setVelocity(v_right)
        self.b_left_motor.setVelocity(v_left)
        self.b_right_motor.setVelocity(v_right)

        self.robot.step(self.timestep)

        observation = self._get_obs()
        dist = observation["goal_info"][0]
        lidar_obs = observation["lidar"]

        terminated = False
        truncated = False
        reward = 0.0

        #if the robot tips over 
        orientation = self.robot_node.getOrientation()
        if orientation[8] < 0.5:
            reward -= 20.0
            terminated = True

        distance_delta = self.previous_distance - dist
        reward += distance_delta * 5.0
        self.previous_distance = dist

        if dist < 0.5:
            reward += 50.0
            terminated = True

        if lidar_obs[1] < 0.3:
            reward -= 15.0
            terminated = True

        info = {}
        
        return observation, reward, terminated, truncated, info
    