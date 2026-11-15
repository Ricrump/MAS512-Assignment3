from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from webots_env import GridWorldEnv

env = GridWorldEnv()

model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048, batch_size=64)

model.learn(total_timesteps=100000)
model.save("PPO_webots_100k")
print("¡Modelo inicial de 100k guardado!")

"""

"""




"""
model = PPO.load("PPO_webots_navigation", print_system_info=True)

plt.figure(figsize=(8, 8))

for test_run in range(5):
    obs, _ = env.reset()
    done = False
    trajectory_x = []
    trajectory_y = []
    
    while not done:
        # Obtener posición actual para la gráfica
        pos = env.gps.getValues()
        trajectory_x.append(pos[0])
        trajectory_y.append(pos[1])
        
        # Predecir acción con el modelo entrenado
        action, _states = model.predict(obs)
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