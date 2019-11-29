from keras import models, layers, optimizers  # Tensorflow 2.0 Backend
import numpy as np
from collections import deque
import random
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DISCOUNT_RATE = 0.99
LEARNING_RATE = 0.001
REPLAY_MEMORY_SIZE = 30000
MINIBATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):
        # Environment
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # DQN Extension
        self.double_q = double_q
        self.per = per
        self.multistep = multistep
        self.n_steps = 1
        # Parameter
        self.gamma = DISCOUNT_RATE
        self.learning_rate = LEARNING_RATE
        self.minibatch_size = MINIBATCH_SIZE
        # Network and Memory
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def _build_network(self):
        network = models.Sequential([
            layers.Dense(32, activation="relu", input_shape=(self.state_size, )),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        network.compile(loss="mse", optimizer=optimizers.Adam(self.learning_rate))
        return network

    def predict(self, state, epsilon):
        if random.random() > epsilon:
            return np.argmax(self.network.predict_on_batch(np.expand_dims(state, axis=0)))
        else:
            return np.random.choice(self.action_size)

    def train_minibatch(self, minibatch):
        states = np.array([experience[0] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        policy = np.array(self.network.predict_on_batch(states))
        target_policy = np.array(self.target_network.predict_on_batch(next_states))
        for idx, experience in enumerate(minibatch):
            _, action, reward, _, done = experience
            policy[idx, action] = reward if done else reward + self.gamma * np.max(target_policy[idx])
        self.network.train_on_batch(x=states, y=policy)

    def learn(self, max_episode: int = 1000):
        episode_record = list()
        episode_rewards = deque(maxlen=100)
        epsilon = EPSILON_START
        epsilon_min = EPSILON_MIN
        epsilon_decay = EPSILON_DECAY

        print("=" * 68)
        print(f"      Double: {self.double_q}\t  Multistep: {self.multistep}/{self.n_steps}\t   PER: {self.per}")
        print("=" * 68)

        for episode in range(max_episode):
            start_time = time.time()
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.predict(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_memory.append((state, action, reward, next_state, done))

                total_reward += reward
                state = next_state

                if len(self.replay_memory) > self.minibatch_size:
                    self.train_minibatch(random.sample(self.replay_memory, self.minibatch_size))

            self.target_network.set_weights(self.network.get_weights())
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            episode_rewards.append(total_reward)
            avg_episode_rewards = np.mean(episode_rewards)
            episode_record.append(avg_episode_rewards)
            episode_time = time.time() - start_time
            print(f"[Episode {episode:5d}] Steps: {-total_reward:3.0f} | Avg reward: {avg_episode_rewards:8.3f} | Time: {episode_time:.3f} secs")

        return episode_record
