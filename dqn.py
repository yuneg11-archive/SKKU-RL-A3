import sys
import numpy as np
import tensorflow as tf  # Tensorflow 2.0
import random
import gym
from collections import deque

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
LEARNING_RATE = 0.001
LEARNING_STARTS = 1000


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.double_q = double_q
        self.per = per
        self.multistep = multistep

        self.n_steps = 1

    def _build_network(self, ):
        # Target network and Local network
        pass

    def predict(self, state):
        # Predict an action from the state based on the policy
        return self.env.action_space.sample()

    def train_minibatch(self, ):
        # Update the policy from mini batch
        pass

    def update_epsilon(self, ):
        # Update epsilon value used in exploration
        pass

    def learn(self, max_episode: int = 1000):
        episode_record = []
        last_100_game_reward = deque(maxlen=100)

        print("=" * 52)
        print(f"Double: {self.double_q}\tMultistep: {self.multistep}/{self.n_steps}\tPER: {self.per}")
        print("=" * 52)

        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0

            # Start episode
            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1

                state = next_state
                step_count += 1

            last_100_game_reward.append(-step_count)
            avg_reward = np.mean(last_100_game_reward)
            episode_record.append(avg_reward)
            print(f"[Episode {episode:>5}] episode steps: {step_count:>5} | avg: {avg_reward:8.3f}")

        return episode_record
