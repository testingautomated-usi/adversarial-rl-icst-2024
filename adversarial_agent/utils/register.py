import highway_env
from gymnasium.envs.registration import register

register(
    id='highwayMA-v0',
    entry_point = 'highway_env.envs:HighwayEnvMA',
)