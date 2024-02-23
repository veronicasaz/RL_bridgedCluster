from gym.envs.registration import register


register(
    id='ThreeBody-v0', # 
    entry_point='bridgedparticles.envs:ThreeBody_env',
    max_episode_steps=1000,
)

