import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    elif args.dataset == 'Avatar':
        image_dataset=init_avatar()
    elif args.dataset == 'TalkingHead':
        image_dataset=init_talking()
    else:
        NotImplementedError


    target_list=image_dataset.label
    dataloader=DataLoader(image_dataset, batch_size=1, shuffle=True)
    output_list=[]
    for image, label in tqdm(dataloader):
        try:
            # face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=image.to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                
                
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if i!=idx_img:
                    pred_list.append([])
                    idx_img=i
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    # print(target_list)
    # print(output_list)
    # acc=sum(target_list==output_list)/len(target_list)
    accsum=0
    true_cnt=0
    false_cnt=0
    true_tot=0
    false_tot=0
    for t, p in zip(target_list, output_list):
        p=round(p)
        accsum+=int(t==p)
        if t == 0:
            false_tot += 1
            false_cnt += int(t==p)
        else:
            true_tot += 1
            true_cnt += int(t==p)
    print(accsum)
    acc=accsum/len(target_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')
    print(f'{args.dataset}| ACC: {acc:.4f}')

    print(f"True ACC: {(true_cnt/true_tot):.4f}")
    print(f"False ACC: {(false_cnt/false_tot):.4f}")






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)

