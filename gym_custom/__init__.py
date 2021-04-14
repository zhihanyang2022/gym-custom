from gym.envs.registration import register

register(
    id='pomdp-mountain-car-v0',
    entry_point='gym_custom.envs:ContinuousMountainCarPomdpEnv',
)

register(
    id='heaven-hell-onehot-ls-v0',
    entry_point='gym_custom.envs:HeavenHellOneHotLSEnv',
)