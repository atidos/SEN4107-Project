import time

import matplotlib.pyplot as plt
import numpy as np

def get_train_data(log):
    with open(log, "r") as f:
        list = f.read().split("\n")

    list.pop(0)
    list.pop(0)
    list.pop(0)
    list.pop(0)
    list.pop(0)

    epochList = []
    train_loss = []
    val_loss = []
    Accuracy = []
    Percision = []
    Recall = []

    for string in list:
        if (string != ""):
            epochList.append(string)

    count = 0
    for x in epochList:
        index = epochList.index(x)
        if index % 3 == 0:
            count += 1
            # print(x.split("loss=")[1])
            train_loss.append(float(x.split("loss=")[1]))
        elif index % 3 == 1:
            # print(x)
            val_loss.append(float(x.split("loss=")[1]))
        else:
            Accuracy.append(float(x.split(" ")[2]))
            Percision.append(float(x.split(" ")[7]))
            Recall.append(float(x.split(" ")[12]))

    arr = np.array(Recall)

    data = {
        "epochs": range(0, count),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": Accuracy,
        "percision": Percision,
        "recall": Recall
    }

    return data

fig = plt.figure()

while True:
    a = get_train_data("checkpoint/logging_data_15_0.001_40_1e-06")
    plt.plot(a["accuracy"])
    plt.plot(a["percision"])
    plt.ylim([0, 100])
    plt.xlim([0, 75])
    plt.draw()
    plt.pause(1)
    fig.clear()

