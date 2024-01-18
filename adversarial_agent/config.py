config_11 = {
  "controlled_vehicles" : 1,
  "observation": {
      "type": "Kinematics",
      "features": ["id", "x", "y", "vx", "vy"],
      "features_range": {
            "x": [-1200, 1200],
      },
      "absolute": True,
      "normalize" : True,
  },
      "lanes_count" : 2,
      "vehicles_count": 1,
      "duration": 30,
      "normalize_reward": True,
      "collision_reward": -1
}

config_ma_2 = {
    "controlled_vehicles" : 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
  "observation": {
        "type": 'MultiAgentObservation',
        "observation_config": {
            "type": "Kinematics",
            "features": ["id", "x", "y", "vx", "vy"],
            "features_range": {
            "x": [-1200, 1200],
            },
            "absolute": True,
            "normalize" : True,
        }  
  },
  "lanes_count" : 2,
  "vehicles_count": 0,
  "reward_vehicle" : 'adversarial',
  "duration": 30,
  "collision_reward": 0,
"normalize_reward": True,
"ego_spacing" : [0.5,2.0],
"simulation_frequency" : 15,
"delta" : 0.0025,
}


config_rt = {
    "controlled_vehicles" : 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
  "observation": {
        "type": 'MultiAgentObservation',
        "observation_config": {
            "type": "Kinematics",
            "features": ["id", "x", "y", "vx", "vy"],
            "features_range": {
            "x": [-1200, 1200],
            "vx": [0,40],
            },
            "absolute": True,
            "normalize" : True,
        }  
  },
  "lanes_count" : 2,
  "vehicles_count": 0,
  "reward_vehicle" : 'adversarial',
  "duration": 30,
  "collision_reward": -1,
"normalize_reward": True,
"ego_spacing" : [0.5,2.0],
"simulation_frequency" : 15,
"delta" : 0.0025,
}