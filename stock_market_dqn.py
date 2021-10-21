# -*- coding: utf-8 -*-
"""
ILESHA SAWARKAR
###

Import Libraries
"""
#Import Libraries

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import pandas as pd
import numpy as np

# CREATE A DQN AGENT

class Agent:
  def __init__(self, state_size, is_eval=False, model_name=""):
    
    # normalized previous days
    self.state_size  = state_size 
    
    # sit, buy, sell
    self.action_size = 3 
    
    self.memory = deque(maxlen=1000)   # Max queue for agent (replay memory)
    
    self.inventory  = []
    self.model_name = model_name
    self.is_eval    = is_eval

    self.gamma   = 0.95 #hyperparameters
    
    self.epsilon = 1.0   #hyperparameters
    self.epsilon_min   = 0.01  #hyperparameters
    self.epsilon_decay = 0.995  #hyperparameters

    self.model = load_model("" + model_name) if is_eval else self._model()

      
  def _model(self):   # Defining the Sequential Neural Network
      
      model = Sequential()
      model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
      model.add(Dense(units=32, activation="relu"))
      model.add(Dense(units=8, activation="relu"))
      
      model.add(Dense(self.action_size, activation="linear"))
      
      model.compile(loss="mse", optimizer=Adam(lr=0.001))

      return model

  def act(self, state):   # Action for the agent
      
      if not self.is_eval and np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)

      options = self.model.predict(state)
      return np.argmax(options[0])

  def expReplay(self, batch_size):  # Replay to check best action
      
      mini_batch = []
      l          = len(self.memory)
      
      for i in range(l - batch_size + 1, l):
          mini_batch.append(self.memory[i])

      for state, action, reward, next_state, done in mini_batch:
          target = reward
          
          if not done:
              # amax = Return the maximum of an array or maximum along an axis
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

          target_f = self.model.predict(state)
          target_f[0][action] = target
          
          self.model.fit(state, target_f, epochs=1, verbose=0)

      if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay

# PREPROCESS THE DATA

import math
import csv

# prints formatted price
def formatPrice(n):
  return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
  
  vec = []
  lines = open("/dataset/" + key + ".csv", "r").read().splitlines()
  
  for line in lines[1:]:
    vec.append(float(line.split(",")[4]))

  return vec

# returns the sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):

  d = t - n + 1
  #print(d,'--> D')

  if d >= 0:
    block = data[d:t + 1] 
  else :
    block=-d * [data[0]] + data[0:t + 1] 
  #print(block,'--> block')
  res = []
  #print(n)
  for i in range(0,n - 1):
    #print(block[i + 1])
    num=block[i + 1] - block[i]
    #print(num)
    res.append(sigmoid(num))
  return np.array([res])

# TRAIN AND BUILD THE MODEL

import sys

if len(sys.argv) != 4:
	print ("Usage: python train.py [stock] [window] [episodes]")
	exit()


stock_name = 'GSPC_Training_Dataset'
window_size = 20 # For faster training
episode_count = 1  # Kernel force restarts so keeping epoch to 1 else 10 will be the best option


agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	print ("Episode " + str(e) + "/" + str(episode_count))  #Epochs
	state = getState(data, 0, window_size + 1)

	total_profit = 0   #starting profit is 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit or hold 
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print ("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	#if e % 10 == 0:
		agent.model.save("/dataset/model_ep" + str(e))
		
# Evaluate the model and agent

import sys
from keras.models import load_model


if len(sys.argv) != 3:
	print ("Usage: python evaluate.py [stock] [model]")
	exit()


stock_name = 'GSPC_Evaluation_Dataset'
model_name = 'model_ep0'

model = load_model("/dataset/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print ("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
