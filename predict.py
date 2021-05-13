from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from test_AncientSites import test_SitesDataset

import warnings
from ResNet import ResNet34, ResNet18, ResNet50
from ancientSiteDataset import AncientSiteDataset

warnings.filterwarnings("ignore")

datasets = test_SitesDataset(transform=transforms.ToTensor())
test_loader = DataLoader(datasets, batch_size=50, shuffle=False, num_workers=4)
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the following two line should be modified for the change of model architecture
#####################################################################################
res1 = ResNet34().to(device0)
res1.load_state_dict(torch.load('models/bestModel/ResNet34Epoch:55.pt'))
res3 = ResNet50().to(device0)
res3.load_state_dict(torch.load('ResStack/ResNet50Epoch:75.pt'))
res2 = ResNet34().to(device0)
res2.load_state_dict(torch.load('ResStack/ResNet34Epoch:45.pt'))

model = [res1,res3,res2]


#####################################################################################

def test(model, test_loader, device):
    m1 = model[0]
    m2 = model[1]
    m3 = model[2]
    m1.eval()
    m2.eval()
    m3.eval()
    pred_label = []
    pred_list = []
    output = {}
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            test_x = batch['image'].to(device)
            test_name = batch['dir']

            pred1 = m1(test_x)
            pred2 = m2(test_x)
            pred3 = m3(test_x)
            
            pred1 = pred1.argmax(dim=1)
            pred2 = pred2.argmax(dim=1)
            pred3 = pred3.argmax(dim=1)
            for p1,p2,p3,name in zip(pred1,pred2, pred3,test_name):
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
                output[name] = out

    return output


output = test(model=model, test_loader=test_loader, device=device0)
names = []
pred = []
for sample in output:
    names.append(sample)
    pred.append(output[sample])
pred_df = pd.DataFrame({'Image name': names, 'predict': pred})
pred_df.to_csv('predict_result.csv', index=False)


