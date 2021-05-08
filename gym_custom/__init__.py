from gym.envs.registration import register

register(
    id='pomdp-mountain-car-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpEnv',
)

register(
    id='pomdp-mountain-car-easy-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpEasyEnv'
)

register(
    id='pomdp-mountain-car-episodic-easy-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpEpisodicEasyEnv'
)

register(
    id='pomdp-mountain-car-episodic-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpEpisodicEnv'
)

register(
    id='pomdp-mountain-car-opt-lower-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpOptLowerEnv'
)

register(
    id='continuous-heaven-hell-opt-lower-v0',
    entry_point='gym_custom.envs:ContinuousHeavenHellOptLower'
)
# register(
#     id='heaven-hell-onehot-ls-v0',
#     entry_point='gym_custom.envs:HeavenHellOneHotLSEnv',
# )