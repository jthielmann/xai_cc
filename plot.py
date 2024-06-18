import matplotlib.pyplot as plt



def plot_train_curves(history, metric_of_interest, batch, lr, date, data_dir):  #metric_of_interest "loss" or "accuracy"
    xi = [5, 10, 15, 20, 25, 30]  #Adapt x-ticks to amount of training episodes
    plt.figure(figsize=(10, 8))

    plt.plot(history["train_episodes"], history["train_" + metric_of_interest], label="Train",      marker=".")
    plt.plot(history["train_episodes"], history["val_"   + metric_of_interest], label="Validation", marker=".")
    plt.ylabel("Pearson's r correlation [n.a.]", fontsize=12)
    plt.title("MSE Loss [n.a]" if metric_of_interest == "loss" else "Pearson's r correlation [n.a.]", fontsize=16)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))

    plt.xlabel("Training Epoch", fontsize=12)
    #plt.xticks(xi)
    plt.margins(x=0)
    plt.legend(fontsize=11)
    plt.xticks(xi, fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()
    plt.savefig(data_dir + date + "_" + "_ST_absolute_multi6_" + str(
        batch) + "_" + lr + "_train-" + metric_of_interest + ".tif")