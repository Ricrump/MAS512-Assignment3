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

        self.obstacle_zones = {
            "OBSTACLE_BLUE": ((-0.2, 0.7), (-0.7, 0.3)),
            "OBSTACLE_RED": ((0.7, 1.5), (-1.4, -0.5)),
        }
        self.obstacle_nodes = {n: self.robot.getFromDef(n) for n in
                               ["OBSTACLE_BLUE", "OBSTACLE_RED", "OBSTACLE_YELLOW", "OBSTACLE_PURPLE"]}
        self.target_node = self.robot.getFromDef("TARGET")
        self.target_initial_position = self.target_node.getPosition()

        self.observation_space = gym.spaces.Dict(
            {
                "lidar": gym.spaces.Box(low=0.0, high=5.0, shape=(24,), dtype=np.float32),
                "goal_info": gym.spaces.Box(low=-np.pi, high=np.inf, shape=(2,), dtype=np.float32),
                "last_action": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.max_wheel_speed = 6.0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.prev_v_left = 0.0
        self.prev_v_right = 0.0
        self.previous_distance = 0.0
        self.t = 0
        self.best_dist = 0.0
        self.stall = 0
        self.idle = 0
        self.idle_ref = [0.0, 0.0]
        self.ranges = None
        self.body_dist = None
        self.parked = [6.0, 6.0, 0.25]

        corner = (-1.6, 1.7)
        goal = (2.09, -1.88)

        def diag(t, h=0.15):
            cx = corner[0] + t * (goal[0] - corner[0])
            cy = corner[1] + t * (goal[1] - corner[1])
            return ((cx - h, cx + h), (cy - h, cy + h))

        self.zones = {
            "d75": [diag(0.75)], "d30": [diag(0.3)], "d20": [diag(0.2)],
            "corner": [((-1.65, -1.45), (1.6, 1.75))],
            "below": [((-1.65, -1.45), (0.1, 0.8))],
            "above": [((-0.5, 0.4), (1.6, 1.75))],
            "mid": [((-1.6, -1.2), (-0.5, 0.5))],
        }
        self.zones["all"] = self.zones["corner"] + self.zones["below"] + self.zones["above"] + self.zones["mid"]
        B, R, Y, P = "OBSTACLE_BLUE", "OBSTACLE_RED", "OBSTACLE_YELLOW", "OBSTACLE_PURPLE"
        self.levels = {
            1: ("d75", ()),
            2: ("d30", (B,)),
            3: ("d30", (B,)),
            4: ("d20", (B, R)),
            5: ("corner", (B, R, Y)),
            6: ("below", (B, R, Y)),
            7: ("above", (B, R, Y)),
            8: ("all", (B, R, Y)),
            9: ("all", (B, R, Y, P)),
        }
        self.center_levels = {2}
        self.level = 9

    def set_level(self, level):
        self.level = int(level)

    def _sample_obstacles(self, names):
        cx, cy = -1.6, 1.7
        gx, gy = 2.09, -1.88
        length = math.hypot(gx - cx, gy - cy)
        ux, uy = (gx - cx) / length, (gy - cy) / length
        px, py = -uy, ux

        def make(t, side):
            return [cx + t * length * ux + side * px, cy + t * length * uy + side * py]

        def max_side(t):
            x, y = cx + t * length * ux, cy + t * length * uy
            lo_x, hi_x, lo_y, hi_y = -2.15 + 0.7, 2.85 - 0.7, -2.71 + 0.7, 2.29 - 0.7
            plus = min((hi_x - x) / px, (hi_y - y) / py)
            minus = min((x - lo_x) / px, (y - lo_y) / py)
            return max(0.0, min(plus, minus))

        def box_dist(p, box):
            (x0, x1), (y0, y1) = box
            return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))

        keepout = [b for z in ("corner", "below", "above", "mid") for b in self.zones[z]]
        n = len(names)
        t0, t1 = 0.27, 0.72
        sep = 1.1 if n <= 3 else 1.0
        for _ in range(300):
            w = (t1 - t0) / n
            ts = [float(self.np_random.uniform(t0 + i * w, t0 + (i + 1) * w)) for i in range(n)]
            sides = [float(self.np_random.uniform(-max_side(t), max_side(t))) for t in ts]
            if self.np_random.random() < 0.5:
                sides[int(self.np_random.integers(n))] = float(self.np_random.uniform(-0.2, 0.2))
            pts = [make(t, s_) for t, s_ in zip(ts, sides)]
            ok = all(math.hypot(p[0] - gx, p[1] - gy) > 1.5 for p in pts)
            ok = ok and all(box_dist(p, b) > 0.9 for p in pts for b in keepout)
            ok = ok and all(math.hypot(a[0] - b[0], a[1] - b[1]) > sep for i, a in enumerate(pts) for b in pts[i + 1:])
            if ok:
                break
        else:
            pts = [make(t0 + (t1 - t0) * i / max(n - 1, 1), 0.6 if i % 2 == 0 else -0.6) for i in range(n)]
        order = self.np_random.permutation(n)
        return {names[int(order[i])]: pts[i] for i in range(n)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed) # IMPORTANT: Must call this first to seed the random number generator

        trans_field = self.robot_node.getField("translation")
        rot_field = self.robot_node.getField("rotation")
        zone_name, active_obstacles = self.levels[self.level]
        for attempt in range(100):
            placed = {}
            if self.level in self.center_levels:
                t = float(self.np_random.uniform(0.5, 0.62))
                side = float(self.np_random.uniform(-0.15, 0.15))
                placed[active_obstacles[0]] = [-1.6 + t * 3.69 + side * 0.697, 1.7 - t * 3.58 + side * 0.718]
            elif len(active_obstacles) <= 2:
                for name, (xr, yr) in self.obstacle_zones.items():
                    if name not in active_obstacles:
                        continue
                    for _ in range(30):
                        pos = [float(self.np_random.uniform(*xr)), float(self.np_random.uniform(*yr))]
                        if all(math.hypot(pos[0] - p[0], pos[1] - p[1]) > 0.8 for p in placed.values()):
                            break
                    placed[name] = pos
            else:
                placed = self._sample_obstacles(list(active_obstacles))
            start = [0.0, 0.0, 0.0]
            boxes = self.zones[zone_name]
            for _ in range(30):
                xr, yr = boxes[int(self.np_random.integers(len(boxes)))]
                start[0] = float(self.np_random.uniform(*xr))
                start[1] = float(self.np_random.uniform(*yr))
                if all(math.hypot(start[0] - p[0], start[1] - p[1]) > 0.8 for p in placed.values()):
                    break
            if all(math.hypot(start[0] - p[0], start[1] - p[1]) > 0.9 for p in placed.values()):
                break
        trans_field.setSFVec3f(start)
        for name in self.obstacle_nodes:
            pos = placed.get(name)
            self.obstacle_nodes[name].getField("translation").setSFVec3f(pos + [0.25] if pos else self.parked)
        yaw = -float(self.np_random.uniform(0.0, math.pi / 2))
        rot_field.setSFRotation([0.0, 0.0, 1.0, yaw])

        self.robot_node.resetPhysics()

        self.f_left_motor.setVelocity(0.0)
        self.f_right_motor.setVelocity(0.0)
        self.b_left_motor.setVelocity(0.0)
        self.b_right_motor.setVelocity(0.0)

        self.prev_v_left = 0.0
        self.prev_v_right = 0.0

        for _ in range(5):
            self.robot.step(self.timestep)

        observation = self._get_obs()
        self.previous_distance = observation["goal_info"][0]
        self.t = 0
        self.best_dist = float(observation["goal_info"][0])
        self.stall = 0
        self.idle = 0
        self.idle_ref = list(self.gps.getValues())[:2]
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
        lidar_data = [5.0] * 24
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
                    self.ranges = np.array(cleaned_lidar)
                    for k in range(4):
                        i = (k * n) // 4 + n // 8 - 1
                        self.ranges[i] = min(self.ranges[i - 1], self.ranges[(i + 1) % n])
                    if self.body_dist is None or len(self.body_dist) != n:
                        ang = (np.arange(n) - n / 2) * 2 * math.pi / n
                        c, sn = np.cos(ang), np.abs(np.sin(ang))
                        with np.errstate(divide="ignore"):
                            tx = np.where(c > 1e-9, 0.05 / c, np.where(c < -1e-9, -0.45 / c, np.inf))
                            ty = np.where(sn > 1e-9, 0.25 / sn, np.inf)
                        self.body_dist = np.minimum(tx, ty)
                    gap = np.clip(self.ranges - self.body_dist, 0.0, 5.0)
                    rolled = np.roll(gap, -(n // 2 - n // 48))
                    lidar_data = [float(c.min()) for c in np.array_split(rolled, 24)]

        observation = {
            "lidar": np.array(lidar_data, dtype=np.float32),
            "goal_info": np.array([dist, heading_error], dtype=np.float32),
            "last_action": np.array([self.prev_v_left, self.prev_v_right], dtype=np.float32) / self.max_wheel_speed,
        }

        return observation

    def step(self, action):
        target_v_left = float(action[0]) * self.max_wheel_speed
        target_v_right = float(action[1]) * self.max_wheel_speed

        # ACTION SMOOTHING (Inertia filter) – yep, Gemini idea
        alpha = 0.7
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

        terminated = False
        truncated = False
        reward = 0.0

        #if the robot tips over
        orientation = self.robot_node.getOrientation()
        if orientation[8] < 0.5:
            reward -= 40.0
            terminated = True

        self.t += 1
        reward -= 0.1

        distance_delta = self.previous_distance - dist
        reward += distance_delta * 30.0
        self.previous_distance = dist

        success = bool(dist < 0.5)
        if success:
            reward += 50.0
            terminated = True

        clearance = 5.0
        if self.ranges is not None:
            ang = (np.arange(len(self.ranges)) - len(self.ranges) / 2) * 2 * math.pi / len(self.ranges)
            px, py = self.ranges * np.cos(ang), self.ranges * np.sin(ang)
            dx = np.maximum(np.maximum(-0.45 - px, px - 0.05), 0.0)
            dy = np.maximum(np.abs(py) - 0.25, 0.0)
            clearance = float(np.min(np.hypot(dx, dy)))
        crash = clearance < 0.04
        if crash:
            reward -= 40.0
            terminated = True

        if dist < self.best_dist - 0.05:
            self.best_dist = float(dist)
            self.stall = 0
        else:
            self.stall += 1
        pos = self.gps.getValues()
        if math.hypot(pos[0] - self.idle_ref[0], pos[1] - self.idle_ref[1]) > 0.15:
            self.idle_ref = [pos[0], pos[1]]
            self.idle = 0
        else:
            self.idle += 1
        if not terminated and self.idle >= 100:
            reward -= 80.0
            terminated = True
        if not terminated and self.stall >= 300:
            reward -= 80.0
            terminated = True

        if not terminated and self.t >= 2000:
            reward -= 80.0
            truncated = True

        if clearance < 0.4:
            reward -= (0.4 - clearance) * 1.0

        info = {"is_success": success}

        return observation, reward, terminated, truncated, info
