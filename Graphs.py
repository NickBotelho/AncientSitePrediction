#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def getTrainLoss(trainLoss,name=""):
    #[[epoch][loss]]
    plt.scatter(trainLoss[0],trainLoss[1])
    plt.plot(trainLoss[0],trainLoss[1])
    plt.title("Train Loss Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Graphs/"+name+"trainLoss.png")
    plt.clf()

def getTestAccuracy(testAccuracy,name=""):
    #[[epoch][accuracy]]
    plt.scatter(testAccuracy[0],testAccuracy[1])
    plt.plot(testAccuracy[0],testAccuracy[1])
    plt.title("Test Accuracy Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("Graphs/"+name+"testAccuracy.png")
    plt.clf()

def getF1(f1Score, name = ""):
    #[[epoch][f1Score]]
    plt.scatter(f1Score[0],f1Score[1])
    plt.plot(f1Score[0],f1Score[1])
    plt.title("f1Score Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("f1Score")
    plt.savefig("Graphs/"+name+"f1Score.png")
    plt.clf()