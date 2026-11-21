from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from webots_env import GridWorldEnv


class CurriculumCallback(BaseCallback):
    thresholds = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.6, 6: 0.8, 7: 0.8, 8: 0.7}

    def __init__(self, window=40):
        super().__init__()
        self.window = window
        self.level, self.results = 1, []

    def _on_training_start(self):
        self.training_env.env_method("set_level", self.level)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.results.append(bool(info.get("is_success", False)))
        if len(self.results) >= self.window:
            rate = sum(self.results[-self.window:]) / self.window
            if self.level < 9 and rate >= self.thresholds[self.level]:
                self.level += 1
                self.training_env.env_method("set_level", self.level)
                self.results = []
                print(f"Curriculum: level {self.level} at step {self.num_timesteps}", flush=True)
            if self.level == 9 and len(self.results) >= 50 and sum(self.results[-50:]) / 50 >= 0.9:
                return False
        return True


env = DummyVecEnv([lambda: Monitor(GridWorldEnv())])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO("MultiInputPolicy", env, verbose=1, n_steps=4096, batch_size=256,
            gamma=0.997, ent_coef=0.001, learning_rate=1e-4, n_epochs=5, target_kl=0.03,
            policy_kwargs=dict(net_arch=[512, 512, 512, 512]))
model.learn(total_timesteps=5000000, callback=CurriculumCallback())

model.save("../../PPO_final")
env.save("../../PPO_final_vecnormalize.pkl")
print("Training done.")
