import numpy as np
import gym

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym.envs.registration import register
import tensorflow as tf
import datetime

register(
    id='track-v1',
    entry_point='turtlely:turtleTrack',
)

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make("track-v1")
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential()
model.add(Flatten(input_shape=(1,2)))
model.add(Dense(16384))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True, callbacks=[tensorboard_callback])