#This file contains:
# - function for creating the A3C-network
# - function for updating parameters
# - function for syncing the parameters
# - function for taking actions

import tensorflow as tf
import gym
import numpy as np
import time

env = gym.make('CartPole-v0')
tf.reset_default_graph()

class ActorCriticNetwork():
    def __init__(self, scope, globalAC=None):
        #split Global_AC from worker_AC for allowing to create the first network. If this split is not made, an error will occur for the pull_*_params
        if scope == 'Global_AC':
            with tf.name_scope(scope):
                self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='state')
                self.actor_params, self.critic_params = self._build_network(scope)[-2:]
        else:
            with tf.name_scope(scope):
                self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='state')
                self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None,], name='action')
                self.target_state_value = tf.placeholder(dtype=tf.float32, shape=[None,1], name='target_value')

                self.action_dist, self.state_value, self.actor_params, self.critic_params = self._build_network(scope)

                temporal_difference = tf.subtract(self.target_state_value, self.state_value, name='temporal_difference')

                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(temporal_difference))

                with tf.name_scope('actor_loss'):
                    #the loss takes the softmax value (between 0-1) and computes the negative log and multiplies this with de temporal difference
                    #Consequently, actions with high certainty and accurate score almost add nothing to the loss (-log(~1)*(y^hat - y ~= 0) = 0
                    #actions with low certainty and better than expected value --> -log(0.1) * (y^hat - y)--> high_value *
                    self.log_prob = -tf.reduce_sum(tf.log(self.action_dist) * tf.one_hot(self.action_placeholder, 2, dtype=tf.float32), axis=1, keepdims=True) #the '2' is for the number of actions that can be taken in the environment of CartPole
                    self.exp_v = self.log_prob * temporal_difference



            self.pull_a_params = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, globalAC.actor_params)]

    def _build_network(self, scope):
        with tf.variable_scope(scope+'/actor_network'):
            layer_1_actor = tf.layers.dense(self.state_placeholder, 150, tf.nn.relu, name='actor_layer_1')
            layer_2_actor = tf.layers.dense(layer_1_actor, 50, tf.nn.relu, name='actor_layer_2')
            layer_3_actor = tf.layers.dense(layer_2_actor, env.action_space.n, name='actor_layer_3')
            action_distribution = tf.nn.softmax(layer_3_actor, name='action_distribution')
        with tf.variable_scope(scope+'/critic_network'):
            layer_1_critic = tf.layers.dense(self.state_placeholder, 100, tf.nn.relu, name='critic_layer_1')
            layer_2_critic = tf.layers.dense(layer_1_critic, 50, tf.nn.relu, name='critic_layer_2')
            state_value = tf.layers.dense(layer_2_critic, 1, name='state_value')

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor_network')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic_network')
            return action_distribution, state_value, actor_params, critic_params
    # def update_global(self):
    #
    # def choose_action(self, state):

Global_AC_Network = ActorCriticNetwork('Global_AC')
Local_AC = ActorCriticNetwork('Local_AC', Global_AC_Network)




with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/stefruinard/Desktop/AI/A3C/graphs/', sess.graph, filename_suffix=str(time.time()))
    sess.run(tf.global_variables_initializer())
    obs = env.reset().reshape(-1,4)
    log_values, exp_values = sess.run([Local_AC.log_prob, Local_AC.exp_v], feed_dict={Local_AC.state_placeholder:obs, Local_AC.action_placeholder:[1], Local_AC.target_state_value:[[10]]})


