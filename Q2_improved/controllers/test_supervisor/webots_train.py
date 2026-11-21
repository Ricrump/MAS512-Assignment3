from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from webots_env import GridWorldEnv
import time

env = GridWorldEnv()






"""
model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=50000)
model.save("PPO_webots_100k")
print("¡Modelo inicial de 100k guardado!")

model = PPO.load("PPO_webots_140k_final", env=env, print_system_info=True)
model.learn(total_timesteps=60000, reset_num_timesteps=False)
model.save("PPO_webots_160k")
print("¡Entrenamiento adicional completado y guardado!")
"""

model = PPO.load("../../PPO_final", print_system_info=True)
norm = VecNormalize.load("../../PPO_final_vecnormalize.pkl", DummyVecEnv([lambda: env]))
norm.training = False

plt.figure(figsize=(8, 8))

for test_run in range(5):
    print(f"Running test {test_run + 1}...")
    obs, _ = env.reset()
    done = False
    trajectory_x = []
    trajectory_y = []
    time.sleep(1)
    while not done:
        # Obtener posición actual para la gráfica
        pos = env.gps.getValues()
        trajectory_x.append(pos[0])
        trajectory_y.append(pos[1])
        
        # Predecir acción con el modelo entrenado
        action, _states = model.predict(norm.normalize_obs(obs), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    plt.plot(trajectory_x, trajectory_y, label=f'Test {test_run + 1}')

plt.title('Trajectories for the 5 autonomous navigation tests (PPO)')
plt.xlabel('Axis X (meters)')
plt.ylabel('Axis Y (meters)')
plt.legend()
plt.grid(True)
plt.savefig('test_trajectories.png')
plt.show()


"""


"""