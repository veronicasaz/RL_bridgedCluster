from gym.envs.registration import register


register(
    id='Cluster_env_hydro-v0', # 
    entry_point='env_hydro.BridgedCluster_env_hydro:Cluster_env_hydro',
    max_episode_steps=1000,
)

