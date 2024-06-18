import torch.nn as nn

from func import get_model, training

# for logging purposes
training(resnet=get_model(),
         data_dir='../Training_Data/',
         epochs=30,
         loss_fn=nn.MSELoss(),
         learning_mode="LLR",
         batch_size=64,
         gene="RUBCNL")
