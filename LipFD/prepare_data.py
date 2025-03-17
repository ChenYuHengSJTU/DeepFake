import os
import cv2
from colorama import init, Fore
import tqdm
import tyro
import numpy as np

def extrace_frame(vid_path, save):
    cap=cv2.VideoCapture(vid_path)
    cnt=0
    flag=0
    tnt=0
    img_list=[]
    vid_name=os.path.basename(vid_path).split('.')[0]
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        # print(frame.size, frame.shape)
        h,w,c=frame.shape
        if h<=0 or w<=0:
            continue
        frame=cv2.resize(frame, (500, 500))
        img_list.append(frame)
        flag+=1
        if flag==5:
            flag=0
            tnt+=1
            image=cv2.hconcat(img_list)
            cv2.imwrite(os.path.join(save, f"{vid_name}_{tnt:04d}.png"), image)
            img_list=[]
        # cnt+=1
    return

def main(src:str='/data3/3DFF/', subdir:str='TalkingHead'):
    real_path=os.path.join(src, subdir, 'video', 'real')
    fake_path=os.path.join(src, subdir, 'video', 'fake')
    real_save=os.path.join(src, subdir, 'image', 'real')
    os.makedirs(real_save, exist_ok=True)
    fake_save=os.path.join(src, subdir, 'image', 'fake')
    os.makedirs(fake_save, exist_ok=True)
    for v in sorted(os.listdir(real_path)):
        print(v)
        extrace_frame(os.path.join(real_path, v), real_save)
    for v in sorted(os.listdir(fake_path)):
        print(v)
        extrace_frame(os.path.join(fake_path, v), fake_save)
    pass

if __name__=="__main__":
    tyro.cli(main)