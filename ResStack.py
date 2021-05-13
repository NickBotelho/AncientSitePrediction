#Imports
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import argparse
import tqdm as tqdm
import time


from DenseNet import DenseNet
from ancientSiteDataset import AncientSiteDataset
from ResNet import ResNet34, ResNet18, ResNet50
from Graphs import getTrainLoss, getTestAccuracy, getF1

def parse_args():
    parser = argparse.ArgumentParser(description='AncientSites Detection')
    parser.add_argument('--batch-size', type=int, default = 128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default = 5,
                        help='Number of epochs to train each model for (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--root-dir', type = str, default = '/local_scratch/COSI149B/Project1/',
                        help='root where train and test folders is on')
    parser.add_argument('--seed', type=int, default = 100,
                        help='Random seed (default: 123)')
    args = parser.parse_args()
    return args
    

def test(model, test_loader, device, roc=None):
    m1 = model[0]
    m2 = model[1]
    m3 = model[2]
    m1.eval()
    m2.eval()
    m3.eval()

    tp,tn,fp,fn,n,p = 1,1,1,1,1,1
    with torch.no_grad():
        for sample in test_loader:
            data = sample['image']
            truth = sample['label']
            data = data.to(device)
            truth = truth.to(device)
            pred1 = m1(data)
            pred2 = m2(data)
            pred3 = m3(data)
            output = []
            pred1 = pred1.argmax(dim=1)
            pred2 = pred2.argmax(dim=1)
            pred3 = pred3.argmax(dim=1)
            for p1,p2,p3 in zip(pred1,pred2, pred3):
                pos = 0
                neg = 0
                if p1 == 1:
                    pos+=1
                else:
                    neg+=1
                if p2 == 1:
                    pos+=1
                else:
                    neg+=1
                if p3 == 1:
                    pos+=1
                else:
                    neg+=1
                out = 1 if pos > neg else 0
                output.append(out)
            
            for pred,truth in zip(output,truth):
                tp = tp+1 if (pred == 1 and truth == 1) else tp
                tn = tn+1 if (pred == 0 and truth == 0) else tn
                fp = fp+1 if (pred == 1 and truth == 0) else fp
                fn = fn+1 if (pred == 0 and truth == 1) else fn
                p = p+1 if (truth == 1) else p
                n = n+1 if (truth == 0) else n
    accuracy = (tp+tn)/(p+n)
    sensitivity = (tp/p) #true positive rate
    specificity = (tn/n) #true negative rate
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = ((precision*recall*2)/(precision+recall))
    print("Confusion Matrix\n"+
    "   0  1\n"+
    "0  {}  {}\n".format(tn, fn)+
    "1  {}  {}".format(fp, tp))
    print("Accuracy = {}%\nSensitivity = {}%\nSpecificity = {}%\nPrecision = {}%\nRecall = {}%\nF1-Score = {}%\n".format(
    accuracy*100, sensitivity*100,specificity*100,precision*100, recall*100, f1*100))
   
   
def test2(model, test_loader, device):
    model.eval()
    tp,tn,fp,fn,n,p = 1,1,1,1,1,1
    with torch.no_grad():
        for sample in test_loader:
            data = sample['image']
            truth = sample['label']
            data = data.to(device)
            truth = truth.to(device)
            output = model(data)
            output = output.argmax(dim = 1)
            for pred,truth in zip(output,truth):
                tp = tp+1 if (pred == 1 and truth == 1) else tp
                tn = tn+1 if (pred == 0 and truth == 0) else tn
                fp = fp+1 if (pred == 1 and truth == 0) else fp
                fn = fn+1 if (pred == 0 and truth == 1) else fn
                p = p+1 if (truth == 1) else p
                n = n+1 if (truth == 0) else n
    accuracy = (tp+tn)/(p+n)
    sensitivity = (tp/p) #true positive rate
    specificity = (tn/n) #true negative rate
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = ((precision*recall*2)/(precision+recall))
    print("Confusion Matrix\n"+
    "   0  1\n"+
    "0  {}  {}\n".format(tn, fn)+
    "1  {}  {}".format(fp, tp))
    print("Accuracy = {}%\nSensitivity = {}%\nSpecificity = {}%\nPrecision = {}%\nRecall = {}%\nF1-Score = {}%\n".format(
    accuracy*100, sensitivity*100,specificity*100,precision*100, recall*100, f1*100))
    


def main():
    
    transform = transforms.Compose([
    
    transforms.ToTensor(),
    
    ])
    dataset = AncientSiteDataset('coordinates_train.txt', args.root_dir, transform)
    
    
    #print("Length of cross_train = {}, length of cross_test= {}".format(len(cross_train), len(cross_test)))
    #trainset, devset = torch.utils.data.random_split(dataset, [45000, 5220])
    
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    res1 = ResNet34().to(device0)
    res1.load_state_dict(torch.load('models/bestModel/ResNet34Epoch:55.pt'))
    res3 = ResNet50().to(device0)
    res3.load_state_dict(torch.load('ResStack/ResNet50Epoch:75.pt'))
    res2 = ResNet34().to(device0)
    res2.load_state_dict(torch.load('ResStack/ResNet34Epoch:45.pt'))

    model = [res1,res3,res2]
        
    i = 0
    cross_train = []
    numCrossNegative = 0
    numCrossPositive = 0
    cross_test = []
    for sample in dataset:
        if i % 10 == 0:
            cross_test.append(sample)
        else:
            cross_train.append(sample)
            if sample['label'] == 0:
                numCrossNegative+=1
            else:
                numCrossPositive+=1
        i+=1
                
                
    crossClassWeights = [1/numCrossNegative, 1/numCrossPositive]
    class_weights = torch.FloatTensor(crossClassWeights).to(device0)
    crossSampleWeights = [0]*len(cross_train)
    loss_function = NN.CrossEntropyLoss(weight = class_weights)
    

    
    
    test_loader = DataLoader(cross_test, batch_size = args.batch_size, shuffle = True, num_workers = 5)
    test(model = model, test_loader = test_loader, device = device0)
  
    test2(model = res1, test_loader = test_loader, device = device0)
    test2(model = res3, test_loader = test_loader, device = device0)
    test2(model = res2, test_loader = test_loader, device = device0)

#model2 so far best with 83% accuracy
if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    main()
    







