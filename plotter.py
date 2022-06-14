import time

import matplotlib.pyplot as plt
import numpy as np
import random

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

def acc(a,b):
    plt.plot(a["accuracy"], label="0.001", color=(1,0,0))
    plt.plot(b["accuracy"], label="0.0001", color=(0,0,1))

    plt.grid()
    plt.legend()
    plt.ylim(0,100)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def loss(a,b):
    plt.plot(a["val_loss"], label="0.001", color=(1, 0, 0))
    plt.plot(b["val_loss"], label="0.0001", color=(0, 0, 1))

    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.show()

plt.title("MiniXception / FER2013 Dataset / Batch Size:15")
a = get_train_data("custom_logs/logging_data_15_0.001_40_1e-06")
b = get_train_data("checkpoint/logging_data_15_0.0001_40_1e-06")
loss(a,b)

