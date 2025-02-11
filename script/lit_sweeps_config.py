from lit_config import lit_config
debug = lit_config["debug"]
if not debug:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": [10, 20, 40, 70]},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01, 0.001]}
            #"learning_rate": {"values": [0.01]}
        },
    }
else:
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mse_loss"},
        "parameters": {
            "epochs": {"values": [1, 2, 3]},
            #"learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
            "learning_rate": {"values": [0.01]}
        },
    }
