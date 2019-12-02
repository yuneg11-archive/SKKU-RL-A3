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
MULTISTEP_LEN = 3
PRIORITY_RATE = 0.6
PRIORITY_ADJUST_START = 0.5
PRIORITY_EPSILON = 0.001

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95


class ReplayMemory:
    def __init__(self, memory_size, multistep_len=0, gamma=0.0, alpha=0.0, beta=1.0):
        # Normal Memory
        self.data_state = deque(maxlen=memory_size)
        self.data_action = deque(maxlen=memory_size)
        self.data_reward = deque(maxlen=memory_size)
        self.data_next_state = deque(maxlen=memory_size)
        self.data_done = deque(maxlen=memory_size)
        self.add = self.multistep_add if multistep_len > 0 else self.single_add
        self.sample_idx = self.per_idx if alpha > 0.0 and beta < 1.0 else self.uniform_idx
        # Multistep Memory
        self.temp_memory = deque(maxlen=multistep_len)
        self.multistep_len = multistep_len
        self.gamma = gamma
        # Prioritized Experience Replay
        self.priority_alpha = deque(maxlen=memory_size)
        self.alpha = alpha  # Suppress priority if alpha == 0.0
        self.beta = beta  # Compensate priority if beta == 1.0

    def __len__(self):
        return len(self.data_state)

    def single_add(self, state, action, reward, next_state, done):
        self.data_state.append(tuple(state))
        self.data_action.append(action)
        self.data_reward.append(reward)
        self.data_next_state.append(tuple(next_state))
        self.data_done.append(done)
        if self.alpha > 0.0 and self.beta < 1.0:
            self.priority_alpha.append(max(self.priority_alpha) if len(self.priority_alpha) > 0 else 1)

    def multistep_add(self, state, action, reward, next_state, done):
        self.temp_memory.append((state, action, reward, next_state, done))
        if done:
            while len(self.temp_memory) > 0:
                self.single_add(*self.multistep_experience())
        elif len(self.temp_memory) == self.multistep_len:
            self.single_add(*self.multistep_experience())

    def multistep_experience(self):
        state, action, reward, _, _ = self.temp_memory[0]
        _, _, _, next_state, done = self.temp_memory[-1]
        reward += sum([(self.gamma ** (idx + 1)) * exp[2] for idx, exp in enumerate(self.temp_memory)])
        self.temp_memory.popleft()
        return state, action, reward, next_state, done

    def uniform_idx(self, sample_size, progress=None):
        idxs = np.random.choice(len(self), sample_size)
        return idxs, None

    def per_idx(self, sample_size, progress):
        priority_alphas = np.array(self.priority_alpha)
        probabilities = priority_alphas / np.sum(priority_alphas)
        idxs = np.random.choice(len(self), sample_size, p=probabilities)

        beta = self.beta + (1 - self.beta) * progress
        weights_raw = np.power(probabilities * len(self), -beta)
        weights = (weights_raw / np.max(weights_raw))[idxs]
        return idxs, weights

    def sample(self, sample_size, progress):
        idxs, weights = self.sample_idx(sample_size, progress)
        states = [self.data_state[idx] for idx in idxs]
        actions = [self.data_action[idx] for idx in idxs]
        rewards = [self.data_reward[idx] for idx in idxs]
        next_states = [self.data_next_state[idx] for idx in idxs]
        dones = [self.data_done[idx] for idx in idxs]

        return idxs, states, actions, rewards, next_states, dones, weights

    def update_priority(self, idxs, priority_alphas):
        for idx, priority_alpha in zip(idxs, priority_alphas):
            self.priority_alpha[idx] = priority_alpha


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
        self.multistep_len = MULTISTEP_LEN if multistep else 0
        # Parameter
        self.gamma = DISCOUNT_RATE
        self.learning_rate = LEARNING_RATE
        self.minibatch_size = MINIBATCH_SIZE
        self.alpha = PRIORITY_RATE if per else 0.0
        self.beta = PRIORITY_ADJUST_START if per else 1.0
        self.priority_epsilon = PRIORITY_EPSILON
        # Network and Memory
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, self.multistep_len, self.gamma, self.alpha, self.beta)

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

    def train_minibatch(self, progress):
        idxs, states, actions, rewards, next_states, dones, weights = self.replay_memory.sample(self.minibatch_size, progress)
        states, next_states = np.array(states), np.array(next_states)

        policies = np.array(self.network.predict_on_batch(states))
        target_policies = self.target_network.predict_on_batch(next_states)

        priority_alphas = list()
        for policy, action, reward, target_policy, done in zip(policies, actions, rewards, target_policies, dones):
            target_action = np.argmax(policy) if self.double_q else np.argmax(target_policy)
            gamma = (self.gamma ** self.multistep_len) if self.multistep else self.gamma
            expectation = reward if done else reward + gamma * target_policy[target_action]
            if self.per:
                loss = expectation - policy[action]
                priority_alphas.append((abs(loss) + self.priority_epsilon) ** self.alpha)
            policy[action] = expectation

        if self.per:
            self.replay_memory.update_priority(idxs, priority_alphas)

        sample_weights = weights if self.per else None
        self.network.train_on_batch(x=states, y=policies, sample_weight=sample_weights)

    def learn(self, max_episode: int = 1000):
        episode_record = list()
        episode_rewards = deque(maxlen=100)
        epsilon = EPSILON_START
        epsilon_min = EPSILON_MIN
        epsilon_decay = EPSILON_DECAY

        print("=" * 68)
        print(f"- Double: {self.double_q}   - Multistep: {self.multistep}/{self.multistep_len}   - PER: {self.per}")
        print("=" * 68)

        for episode in range(max_episode):
            start_time = time.time()
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.predict(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_memory.add(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if len(self.replay_memory) > self.minibatch_size:
                    self.train_minibatch(episode / max_episode)

            self.target_network.set_weights(self.network.get_weights())
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            episode_rewards.append(total_reward)
            avg_episode_rewards = np.mean(episode_rewards)
            episode_record.append(avg_episode_rewards)
            episode_time = time.time() - start_time
            print(f"[Episode {episode:5d}] Steps: {-total_reward:3.0f} | Avg reward: {avg_episode_rewards:8.3f} | Time: {episode_time:.3f} secs")

        return episode_record
