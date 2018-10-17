import gym
import threading
import multiprocessing as mp
import gym
import tensorflow as tf
import numpy as np
from tensorflow.python.training.coordinator import Coordinator

num_workers = mp.cpu_count()
global_memory = []
workers = []


class Worker():
    def __init__(self, name):
        self.name = name
        self.env = gym.make('CartPole-v0')

    def work(self):
        for i in range(5):
            obs = self.env.reset()
            memory_worker = []
            eps_r = 0
            while True:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)
                memory_worker.append((obs, action, reward, next_obs))

                if done:
                    next_obs = self.env.reset()
                    print('I am working:', self.name, 'eps_reward: ', eps_r)
                    global_memory.append(memory_worker)
                    break

                eps_r += reward
                obs = next_obs


for i in range(num_workers):
    i_name = 'W_%i' % i
    workers.append(Worker(i_name))

with tf.Session() as sess:
    coordinator = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    worker_threads = []

    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coordinator.join(worker_threads)
