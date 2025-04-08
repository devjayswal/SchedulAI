import numpy as np
import gym
from gym import spaces

class TimetableEnv(gym.Env):
    def __init__(self, num_courses, num_slots, num_classrooms):
        super(TimetableEnv, self).__init__()

        # Environment parameters
        self.num_courses = num_courses
        self.num_slots = num_slots
        self.num_classrooms = num_classrooms
        
        # Define action space: (course_index, time_slot, classroom)
        self.action_space = spaces.MultiDiscrete([num_courses, num_slots, num_classrooms])
        
        # Define observation space: A flattened 2D timetable representation
        self.observation_space = spaces.Box(
            low=0, high=num_courses, shape=(num_slots * num_classrooms,), dtype=np.int32
        )

        # Initialize the timetable (empty state)
        self.reset()

    def step(self, action):
        course, time_slot, classroom = action
        reward = 0
        done = False
        
        # Ensure valid action
        if self.state[time_slot, classroom] == 0:
            self.state[time_slot, classroom] = course
            reward += 1  # Reward for a valid placement
        else:
            reward -= 10  # Penalize conflicts

        # Check if scheduling is complete
        if np.count_nonzero(self.state) == self.num_courses:
            done = True

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.num_slots, self.num_classrooms), dtype=np.int32)
        return self.state.flatten()

    def render(self, mode='human'):
        print(self.state)
