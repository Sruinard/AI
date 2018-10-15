
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorboard
import collections
import cv2
import time


# In[21]:

class DenseModel():
    def __init__(self, name_scope):
        self.name_scope = name_scope
        self.model(self.name_scope)
#         self.graphkeys = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        
    def model(self, name_scope):
        #create prediction model
        with tf.variable_scope(name_scope): 
            self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None],name='Target_placeholder')
            self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='State_placeholder') #(batch_size, Pixel_w, Pixel_h, n_frames)
            self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='Action_placeholder') 
            
            #conv_layers
            layer1 = tf.layers.dense(inputs=self.state_placeholder, units=32, activation=tf.nn.relu, name='layer1')
            layer2 = tf.layers.dense(inputs=layer1, units=64, activation=tf.nn.relu, name='layer2')
            layer3 = tf.layers.dense(inputs=layer2, units=32, activation=tf.nn.relu, name='layer3')
            self.Q_value_predictions = tf.layers.dense(inputs=layer2, units=2,) #self.predictions gives the Q-value per action
            
            self.Q_value_predictions_flatten = tf.reshape(self.Q_value_predictions, shape=[1,-1], )
            #select only the Q_values of actions_taken  
            self.action_predictions = tf.gather(self.Q_value_predictions_flatten, self.action_placeholder, axis=1)[0]
            
            #compute_loss
            self.losses = tf.squared_difference(self.target_placeholder, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses, name='loss')
            
            #optimizer
            self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.loss)
            
            


# In[22]:

def store_experience(memory, state, action, reward, done, next_state):
        #receive stacked state/next_state
        memory.append((state, action, reward, done, next_state))


# In[23]:

class Network():
    
    def __init__(self, sess, epsilon_start, epsilon_n_reductions, epsilon_min, env, experience_limit, batch_size, gamma, learning_rate, replace_target_pointer, name_scope_target, name_scope_prediction, save_iter=1000):
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_reduce = (epsilon_start - epsilon_min) / epsilon_n_reductions
        self.epsilon = epsilon_start
        self.env = env
        self.memory = collections.deque(maxlen=experience_limit)
        self.sess = sess
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.replace_target_pointer = replace_target_pointer
        self.q_predict = DenseModel(name_scope_prediction)
        self.q_target = DenseModel(name_scope_target)
        self.prediction_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name_scope_prediction)
        self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name_scope_target)
        self.saver = tf.train.Saver() 
        self.save_iter = save_iter
        
    def prediction(self, state, model):
        #model = self.q_predict or self.q_target
        return self.sess.run(model.Q_value_predictions, feed_dict={model.state_placeholder:state})
    
    def target(self, model, states, actions, rewards, dones, next_states):
        self.next_states_values = self.sess.run(tf.reduce_max(model.Q_value_predictions, axis=1),feed_dict={model.state_placeholder:next_states}) #compute max expected Q-value
        self.next_states_values[dones] = 0 #if episode is done, no discounted Q value (important for convergence)
        self.target_state_action_values = self.next_states_values * self.gamma + rewards  #this is target_value (y)
        return self.target_state_action_values
    
    def optimize(self, states, actions, targets):
        rows = np.arange(self.batch_size)
        rows = rows*2
        actions = actions + rows 
        
        feed_dict = {self.q_predict.target_placeholder:targets, self.q_predict.state_placeholder:states, self.q_predict.action_placeholder:actions}
        _, loss = self.sess.run([self.q_predict.train_op, self.q_predict.loss], feed_dict=feed_dict)
        return loss
    
    def action_epsilon(self, state, model):
        if np.random.uniform(low=0., high=1.) < self.epsilon:
            action = env.action_space.sample()
        else:
            Q_values_for_given_obs = self.prediction(state, model)
            action = np.argmax(Q_values_for_given_obs, axis=1)[0]
        self.epsilon = self.epsilon - self.epsilon_reduce if self.epsilon > self.epsilon_min else self.epsilon_min
        return action   
    
    def update_target_params(self, target_params, estimator_params):
        assign_operation = [tf.assign(t,e) for t,e in zip(target_params, estimator_params)]
        return self.sess.run(assign_operation)
        
#     def replace_target_parameters(self, global_counter):
#         if global_counter % self.replace_target_pointer == 0:
        


# In[24]:

def select_random_batch(memory, batch_size):
    #select a random batch and create arrays of the different components (states, actions, rewards, dones, next_states)

    indices = np.random.choice(len(memory), batch_size)
    batch = [memory[i] for i in indices]
    states = np.array([[[i][0][0] for i in batch]])
    states = np.reshape(states, newshape=(batch_size,4))
    next_states = np.array([[i][0][4] for i in batch])
    next_states = np.reshape(next_states, newshape=(batch_size,4))
    dones = np.array([[i][0][3] for i in batch])
    actions = np.array([[i][0][1] for i in batch])
    rewards = np.array([[i][0][2] for i in batch])
    return batch, states, actions, rewards, dones, next_states      


# In[25]:

tf.reset_default_graph()

loss_values = tf.placeholder(dtype=tf.float32, shape=[])
reward_mean_values = tf.placeholder(dtype=tf.float32, shape=[])
reward_bound_values = tf.placeholder(dtype=tf.float32, shape=[])

loss_mapping = tf.summary.scalar('loss', loss_values)
reward_mean_mapping = tf.summary.scalar('mean_reward', reward_mean_values)
reward_bound_mapping = tf.summary.scalar('bound_reward', reward_bound_values)
episode_reward_list = []
loss_reward_list =[]
with tf.Session() as sess:
    
    s_time = time.time()
    env = gym.make('CartPole-v0')
    global_counter = 0
    DQN = Network(env=env, epsilon_start=1.0, epsilon_min=0.02, epsilon_n_reductions=20000, sess=sess, experience_limit=4000, batch_size=64, gamma=0.99, learning_rate=1e-4, replace_target_pointer=250, name_scope_prediction='prediction', name_scope_target='target', save_iter=500)
    n_episodes = 10000
    episode_rewards_buffer = collections.deque(maxlen=50)
    loss_buffer = []
    
    
    writer = tf.summary.FileWriter("/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/", sess.graph)
    
    sess.run(tf.global_variables_initializer())
    for episode in range(n_episodes):
        s_start = time.time()
        state = env.reset()
        state = state.reshape(1,4)
        episode_reward = 0.0
        
        
        while True:
            
            action = DQN.action_epsilon(model=DQN.q_predict, state=state)
            state_, reward, done, _ = env.step(action)
            store_experience(memory=DQN.memory, action=action, done=done, next_state=state_, reward=reward, state=state)
            if len(DQN.memory) < (DQN.memory.maxlen):
                if done:
                    break
                continue
                
            
            #obtain a batch from memory
            
            #env.render()
            #compute target labels and perform an optimization_step
            
#             if global_counter > DQN.memory.maxlen:
                
            
                
                #loss_buffer.append(loss)
            if global_counter > DQN.memory.maxlen and global_counter % DQN.replace_target_pointer == 0:
                print("TARGET REPLACED")
                DQN.update_target_params(estimator_params=DQN.prediction_params, target_params=DQN.target_params)

            if global_counter % DQN.save_iter ==0:
                DQN.saver.save(sess, '/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/Temp_save', global_step=global_counter)
#             update target_network_parameters
           
            episode_reward += reward
            if done:
                break
            
            
            
            state=state_
            state=state.reshape(-1,4)
            global_counter+=1 
       
    
        if global_counter > DQN.memory.maxlen:
            batch, states, actions, rewards, dones, next_states = select_random_batch(DQN.memory, DQN.batch_size)
            #print("OPTIMIZIE")
            q_target_values = DQN.target(actions=actions, dones=dones, model=DQN.q_predict, next_states=next_states, rewards=rewards, states=states)
            loss = DQN.optimize(actions=actions, states=states, targets=q_target_values) #this step also runs a optimization step

            loss_reward_list.append(loss)
            loss_value = sess.run(loss_mapping, feed_dict={loss_values:loss})
            writer.add_summary(global_step=global_counter, summary=loss_value)
        
        s_end = time.time()       
        print('Episode_reward: ', episode_reward, 'Time:', (s_end - s_start), 'Episode_number: ', episode, 'Epsilon:', DQN.epsilon)
        episode_rewards_buffer.append(episode_reward)
        episode_reward_list.append(episode_reward)
        reward_value = sess.run(reward_mean_mapping, feed_dict={reward_mean_values:episode_reward})
        writer.add_summary(global_step=global_counter, summary=reward_value)
        
        merged = tf.summary.merge_all()
        
        if np.mean(episode_rewards_buffer) > 195:
            DQN.saver.save(sess, '/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/final_summary', global_step=global_counter)
            break
        


# In[35]:

get_ipython().system("tensorboard --logdir='/Users/stefruinard/Desktop/RL_models/DQN/Save_sess/'")


# In[29]:

import matplotlib.pyplot as plt
plt.plot(np.arange(2338),episode_reward_list, )
plt.show()


# In[ ]:




# In[ ]:



