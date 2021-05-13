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
    

def train(model, train_loader, optimizer, epoch, device, loss_function):
    model.train()
    lossSum = 0
    for batch_idx, sample in enumerate(train_loader): #get batch
        data = sample["image"]
        truth = torch.tensor(sample['label'], dtype=torch.long)
        data = data.to(device)
        truth = truth.to(device)

        output = model(data) #pass batch

        optimizer.zero_grad() #zero gradients 
        loss = loss_function(output, truth) #calculate loss
        loss.backward() #calculate gradient

        optimizer.step() #update weights
        lossSum += loss.item()
        if (batch_idx + 1) % 100 == 0:
                print('loss: at epoch', epoch, 'iter', batch_idx, loss.item())
    return lossSum
   
def test(model, test_loader, device):
    model.eval()
    tp,tn,fp,fn,n,p = 1,1,1,1,1,1 #1 to avoid any divide by 0 errors
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
    return [accuracy,f1]


def main():
    transform = transforms.Compose([
    transforms.Resize((129,129)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    
    ])

    dataset = AncientSiteDataset('coordinates_train.txt', args.root_dir, transform)  
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = {
        "ResNet50" : ResNet50(),   
    }   
    for modelName in models:
        initial_lr = args.lr
        model = models[modelName].to(device0)

       #Create Cross validation sets       
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
                    
        #calculate weights based on occurences in the dataset          
        crossClassWeights = [1/numCrossNegative, 1/numCrossPositive]
        class_weights = torch.FloatTensor(crossClassWeights).to(device0)
       
        loss_function = NN.CrossEntropyLoss(weight = class_weights)
        

        train_loader = DataLoader(cross_train, batch_size = args.batch_size, num_workers = 5,
        shuffle = True)
        test_loader = DataLoader(cross_test, batch_size = args.batch_size, shuffle = True, num_workers = 5)
        
        optimizer = optim.Adam(model.parameters(), lr = initial_lr)
        start = time.time()

        #These lists will be used to create graphs
        trainLoss = [[],[]]
        testAccuracy = [[],[]]
        f1Score = [[],[]]
        for epoch in range(args.num_epochs):
            if ((epoch + 1) % 10) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
            
                
            loss = train(model = model, train_loader = train_loader,
            optimizer = optimizer, epoch = epoch, device = device0,
            loss_function = loss_function)

            acc = test(model = model, test_loader = test_loader, device = device0)
            trainLoss[0].append(epoch)
            trainLoss[1].append(loss)
            testAccuracy[0].append(epoch)
            testAccuracy[1].append(acc[0])
            f1Score[0].append(epoch)
            f1Score[1].append(acc[1])
        #call graph functions
        getTrainLoss(trainLoss)
        getTestAccuracy(testAccuracy)
        getF1(f1Score)
        end = time.time()
        print("This took {test} minutes to run {name}".format(test=(start-end)/60, name =modelName))
        torch.save(model.state_dict(), "ResStack/{}Epoch:{}.pt".format(modelName,args.num_epochs))
    
    

#model2 so far best with 83% accuracy
if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    main()
    







