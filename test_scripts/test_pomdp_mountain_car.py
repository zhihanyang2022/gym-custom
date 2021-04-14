import gym

env = gym.make('gym_custom:pomdp-mountain-car-v0')

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
        next_state, reward, done, _ = env.step([-5])
        episode_length += 1
        env.render()
    print(episode_length)