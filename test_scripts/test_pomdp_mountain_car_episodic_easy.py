import gym
from gym.wrappers import TimeLimit

env = TimeLimit(gym.make('gym_custom:pomdp-mountain-car-episodic-easy-v0'), max_episode_steps=15)

print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high)

rewards = []
for i in range(1):
    state = env.reset()
    done = False
    episode_length = 0
    while not done:
        print(state[-1])
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        episode_length += 1
        state = next_state
    print(episode_length)