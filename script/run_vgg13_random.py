import torch.nn as nn
from train import training
from model import get_vgg13_random
import os
import torch.optim as optim
model_save_dir = "../models/vgg13/random_weights/"
os.makedirs(model_save_dir, exist_ok=True)
vgg = get_vgg13_random()
learning_rate = 0.00005
freeze_pretrained = False
weight_decay = 0.005

optimizer = optim.SGD([{"params": vgg.gene1.parameters(), "lr": learning_rate}], weight_decay=weight_decay)

print("base vgg model, model.gene1.eval(), freeze pretrained, no decay")


training(model=vgg,
         data_dir='../Training_Data/',
         model_save_dir=model_save_dir,
         epochs=40,
         loss_fn=nn.MSELoss(),
         optimizer=optimizer,
         learning_rate=learning_rate,
         batch_size=64,
         gene="RUBCNL",
         freeze_pretrained=freeze_pretrained)

