import gym
import gym_custom

env = gym.make('pomdp-mountain-car-opt-lower-v0')

print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high)

rewards = []
for i in range(100):
    state = env.reset()
    done = False
    episode_length = 0
    while not done:
        action = env.action_space.sample()
        # next_state, reward, done, _ = env.step([-15])
        next_state, reward, done, _ = env.step(action)
        episode_length += 1
        env.render()
    print(episode_length)