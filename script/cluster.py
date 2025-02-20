# check if main
if __name__ == '__main__':
    debug = True
    if debug:
        import torch
        import numpy
        import sys
        import torchvision
        sys.path.insert(0, '..')
        print("python version:", sys.version)
        print("numpy version:", numpy.version.version)
        print("torch version:", torch.__version__)
        print("torchvision version:", torchvision.__version__)

    model_base_dir = "../models/"
    model_path = "crc_base_RUBCNL_train_normed/ResNet_ep_10_lr_0.01_resnet50random_MSELoss_False_crc_base_32_RUBCNL/"
    model_name = "best_model.pth"
    config_name = "config.json"

    from script.model.lit_model import load_model
    import torch
    import json

    f = open(model_base_dir + model_path + "/config.json", "r")
    str_config = json.load(f)
    print(str_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_base_dir + model_path + model_name, str_config).to(device)
    from script.xai.cluster_functions import cluster
    cluster(model=model, data_dir = str_config["data_dir"], samples = str_config["val_samples"], genes = str_config["genes"], out_dir= "../crp_out/" + model_path, debug=debug)
