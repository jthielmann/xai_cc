import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../models/resnet18/BMP4_random/train_history.csv")
print(df.head())

plt.matshow(df)
plt.show()