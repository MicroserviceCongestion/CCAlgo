# adaptive_qps.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import psutil
import requests
import random

# 设置 TensorFlow 日志级别为 ERROR
tf.get_logger().setLevel('ERROR')
NUM = 2

# 模拟从环境中获取当前的 CPU 使用率和 QPS
def get_current_cpu_usage():
    # 这里应替换为从实际环境获取 CPU 使用率的逻辑
    return psutil.cpu_percent(interval=1)

class AdaptiveRateLimitEnv:
    def __init__(self, c_free, T, N):
        self.c_free = c_free
        self.current_cpu = get_current_cpu_usage()
        self.current_qps = 0
        self.limit = 100  # 初始QPS限制
        self.T = T  # CPU使用率的阈值
        self.N = N  # 状态中包含的性能指标对数量
        self.memory_S = []  # 存储性能指标对

    def reset(self, qps):
        self.current_cpu = get_current_cpu_usage()
        self.current_qps = qps
        self.limit = 100
        self.memory_S.clear()  # 清空历史数据
        self.prepopulate_interactions()
        return self.update_state()  # 返回初始状态

    def prepopulate_interactions(self):
        for _ in range(10):  # 与环境进行10次交互
            state, _, _ = self.step(0)  # 动作a固定为0
            self.limit = self.current_qps
            count_qps = self.current_qps
            if self.current_qps > self.limit:
                count_qps = self.limit

            self.current_cpu = get_current_cpu_usage()
            self.memory_S.append((self.current_qps, self.current_cpu))

    def step(self, action):
        self.limit = self.current_qps * action  # 动作是一个比例，用于计算新的QPS限制阈值
        count_qps = self.current_qps
        if self.current_qps > self.limit:
            count_qps = self.limit

        self.current_cpu = get_current_cpu_usage()  # 从环境中获取实际的 CPU 使用率
        self.collect_performance_metric(count_qps, self.current_cpu)
        state = self.update_state()
        reward = self.calculate_reward(action)

        return state, reward, False

    def collect_performance_metric(self, qps, cpu_usage):
        qps = float(qps) if isinstance(qps, np.ndarray) else qps
        cpu_usage = float(cpu_usage) if isinstance(cpu_usage, np.ndarray) else cpu_usage
        self.memory_S.append((qps, cpu_usage))

        if len(self.memory_S) > self.N:
            self.memory_S.pop(0)

    def update_state(self):
        padded = [(0, self.c_free)] * (self.N - len(self.memory_S))
        state = padded + self.memory_S[-(self.N):]
        return np.array(state).flatten()  # 将状态展平以便输入到网络中

    def calculate_reward(self, action):
        average_qps = np.mean([qps for qps, _ in self.memory_S])
        if self.current_cpu <= self.T:
            reward = average_qps * action
        else:
            reward = -np.exp((self.current_cpu - self.T) / 10)

        return reward

class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer_R = deque(maxlen=200)  # 初始化R列表
        self.memory = np.zeros((200, state_size * 2 * NUM + action_size + 1), dtype=np.float32)
        self.pointer = 0

        self.actor = self.create_actor(state_size * NUM, 1)  # 假设动作空间是1维
        self.critic = self.create_critic(state_size * NUM, 1)
        self.target_actor = self.create_actor(state_size * NUM, 1)
        self.target_critic = self.create_critic(state_size * NUM, 1)
        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.initial_noise_scale = 1.0  # 初始噪声比例
        self.noise_scale = self.initial_noise_scale  # 当前噪声比例
        self.noise_decay = 0.995  # 噪声衰减率

        self.gamma = 0.99
        self.tau = 0.005

    def create_actor(self, state_size, action_size):
        model = tf.keras.Sequential([
            layers.Dense(4, activation='relu', input_shape=(state_size,)),
            layers.Dense(4, activation='relu'),
            layers.Dense(action_size, activation='tanh')  # 使用tanh激活函数来输出动作值
        ])
        model.add(layers.Lambda(lambda x: (x + 1) / 2))  # 调整输出范围到[0, 1]
        return model

    def create_critic(self, state_size, action_size):
        state_input = layers.Input(shape=(state_size,))
        action_input = layers.Input(shape=(action_size,))
        concat = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(4, activation='relu')(concat)
        x = layers.Dense(4, activation='relu')(x)
        x = layers.Dense(1)(x)  # 输出单一的Q值
        return tf.keras.Model([state_input, action_input], x)

    def reset_noise(self):
        self.noise_scale = self.initial_noise_scale  # 重置噪声比例到初始值

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size * NUM])
        action = self.actor.predict(state)[0]
        noise = self.noise_scale * np.random.randn(self.action_size)
        action = np.clip(action + noise, 0, 1)  # 保持动作在[0, 1]之间
        self.noise_scale *= self.noise_decay  # 更新噪声比例
        return action

    def store_transition(self, s, a, r, s_):
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % 200  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        if self.pointer < 200:
            return

        indices = np.random.choice(200, size=64)
        datas = self.memory[indices, :]
        state = datas[:, :self.state_size * NUM]
        action = datas[:, self.state_size * NUM:self.state_size * NUM + self.action_size]
        reward = datas[:, -self.state_size * NUM - 1:-self.state_size * NUM]
        next_state = datas[:, -self.state_size * NUM:]
        state = np.reshape(state, [-1, self.state_size * NUM])
        next_state = np.reshape(next_state, [-1, self.state_size * NUM])

        with tf.GradientTape() as tape:
            target_action = self.target_actor(next_state)
            future_q = self.target_critic([next_state, target_action])
            current_q = self.critic([state, action])
            td_targets = reward + self.gamma * future_q
            loss = tf.losses.mean_squared_error(td_targets, current_q)

        critic_grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state)
            q_values = self.critic([state, actions])
            actor_loss = -tf.reduce_mean(q_values)  # 求最大化Q值，即最小化负Q值

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic.variables, self.tau)

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

class AdaptiveQPSHandler:
    def __init__(self):
        self.env = AdaptiveRateLimitEnv(c_free=10, T=60, N=NUM)
        self.agent = DDPGAgent(state_size=2, action_size=1)
        self.state = None

    def reset(self, qps):
        self.state = self.env.reset(qps)
        self.agent.reset_noise()

    def get_max_qps(self, qps):
        self.env.current_qps = qps  # 更新环境中的 QPS
        action = self.agent.act(self.state)
        next_state, reward, _ = self.env.step(action)
        self.agent.store_transition(self.state, action, reward, next_state)
        self.agent.learn()
        self.state = next_state
        return self.env.limit[0]