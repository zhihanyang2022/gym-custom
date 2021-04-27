import gym
from gym.wrappers import TimeLimit

env = TimeLimit(gym.make('gym_custom:pomdp-mountain-car-episodic-easy-v0'), max_episode_steps=100)

print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high)

rewards = []
for i in range(1):
    state = env.reset()
    done = False
    episode_length = 0
    while not done:
        action = env.action_space.sample()
        # next_state, reward, done, _ = env.step([-100])
        next_state, reward, done, _ = env.step(action)
        print(next_state[-1], reward)
        episode_length += 1
        # env.render()
    print(episode_length)