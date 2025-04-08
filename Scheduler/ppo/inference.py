import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import TimetableEnv
from model import load_model

# Create the environment
env = make_vec_env(lambda: TimetableEnv(num_courses=5, num_slots=6, num_classrooms=3), n_envs=1)

# Load the trained model
model = load_model(env, "ppo_timetable.pth")

# Run inference
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
