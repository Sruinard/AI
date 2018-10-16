import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorboard
import collections
import cv2
import time

sess = tf.Session()
restorer = tf.train.import_meta_graph('/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/final_summary-147479.meta')
restorer.restore(sess, tf.train.latest_checkpoint('/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/'))
graph = tf.get_default_graph()
# print(graph.get_operations())
placeholder_state = graph.get_tensor_by_name('prediction/State_placeholder:0')
predicting_tensor = graph.get_tensor_by_name(name='prediction/dense/BiasAdd:0')
arg_max = tf.argmax(predicting_tensor, axis=1)
env = gym.make('CartPole-v0')
global_counter = 0
n_episodes = 3000
episode_rewards_buffer = collections.deque(maxlen=15)


for episode in range(n_episodes):
    s_start = time.time()
    state = env.reset()
    state = state.reshape(1,4)
    episode_reward = 0.0


    while True:
        action = sess.run(arg_max,feed_dict={placeholder_state:state})[0]
        state_, reward, done, _ = env.step(action)


        #obtain a batch from memory

        env.render()


        episode_reward += reward
        if done:
            break



        state=state_
        state=state.reshape(-1,4)
        global_counter+=1 
    episode_rewards_buffer.append(episode_reward)
    print(episode_reward)
    