import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import TimetableEnv
from model import save_model

# Create and wrap the environment
env = make_vec_env(lambda: TimetableEnv(num_courses=5, num_slots=6, num_classrooms=3), n_envs=1)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
save_model(model, "ppo_timetable.pth")
