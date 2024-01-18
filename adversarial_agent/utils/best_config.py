config_11 = {
  "controlled_vehicles" : 1,
  "observation": {
      "type": "Kinematics",
      "features": ["id", "x", "y", "vx", "vy"],
      "absolute": True,
      "normalize" : True,

  }
}

config_ma = {
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
            "absolute": True,
            "normalize" : True,
        }  
  }
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
            "absolute": True,
            "normalize" : True,
        }  
  },
  "reward_vehicle" : 'adversarial',
}