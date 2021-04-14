import gym
import numpy as np

env = gym.make('gym_custom:heaven-hell-onehot-ls-v0')
env.set_memory(1)

obs = env.reset()
done = False
while not done:
    print(np.argmax(obs[:10]), obs[10])
    action = env.action_space.sample()
    print(action)
    next_obs, reward, done, _ = env.step(action)
    print(reward, done)
    obs = next_obs

