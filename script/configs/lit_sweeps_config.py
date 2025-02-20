from script.configs.lit_config import lit_config
debug = lit_config["debug"]
if not debug:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": [40]},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01]},
            "bins": {"values": [1,3,5,7,9,10,15,50]}
            #"learning_rate": {"values": [0.01]}
        },
    }
else:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": [2]},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01]},
            "bins": {"values": [1, 3, 5, 7, 9, 10, 15, 50]}
        },
    }
