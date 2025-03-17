import os
import csv
import tyro
from tqdm import tqdm
from colorama import init, Fore
from random import random
init(autoreset=True)
import cv2

# some video's format cannot usr GETID to get frame count
# def get_frame_count(vid_path):
#     cap=cv2.VideoCapture(vid_path)
#     cnt=0
#     while True:
#         ret, frame=cap.read()
#         if not ret:
#             break
#         cnt+=1
#     cap.release()
#     return cnt

def get_frame_count(vid_path:str):
    subdirs=vid_path[1:].split('/')
    # print(len(subdirs))
    subdirs[-1]=subdirs[-1].split('.')[0]
    image_dir=os.path.join('/', *subdirs[:2],'extraction',*subdirs[2:-2],subdirs[-1])
    if not os.path.exists(image_dir):
        print(f'{image_dir} not exist')
        exit(2)
    imgs=os.listdir(image_dir)
    # cnt=0
    # for im in imgs:
    #     tmp=cv2.imread(os.path.join(image_dir, im))
    #     if tmp is None:
    #         # print(Fore.RED + f'{os.path.join(image_dir, im)} invalid')
    #         continue
    #     cnt+=1
    return len(imgs), image_dir
    # return cnt, image_dir
    # pass

def get_split():
    rand_val=random()
    if rand_val<0.7:
        return 0
    elif rand_val<0.8:
        return 1
    else:
        return 2


def main(src_folder:str='/data3/FaceForensics++', compression:str='c23'):
    fake_src=os.path.join(src_folder, 'manipulated_sequences')
    real_src=os.path.join(src_folder, 'original_sequences', 'youtube', compression, 'videos')
    fake_types=sorted(os.listdir(fake_src))
    with open('train.txt', 'w+') as f0, open('test.txt', 'w+') as f1, open('val.txt', 'w+') as f2:
        train=csv.writer(f0)
        test=csv.writer(f1)
        val=csv.writer(f2)
        for fk in tqdm(fake_types):
            if fk=='DeepFakeDetection':
                continue
            fake_videos_path=os.path.join(fake_src, fk, compression, 'videos')
            print(Fore.GREEN+fk)
            vpar=tqdm(sorted(os.listdir(fake_videos_path)))
            for v in vpar:
                vpar.set_description(v)
                vid_path=os.path.join(fake_videos_path, v)
                frame_cnt, image_dir=get_frame_count(vid_path)
                split=get_split()
                # vid_path=vid_path.split('.')[0]
                match split:
                    case 0: train.writerow([image_dir, 1, frame_cnt, 1])
                    case 1: test.writerow([image_dir, 1, frame_cnt, 1])
                    case 2: val.writerow([image_dir, 1, frame_cnt, 1])

        vpar=tqdm(sorted(os.listdir(real_src)))
        for real in vpar:
            vpar.set_description(real)
            vid_path=os.path.join(real_src, real)
            frame_cnt, image_dir=get_frame_count(vid_path)
            split=get_split()
            vid_path=vid_path.split('.')[0]
            match split:
                case 0: train.writerow([image_dir, 1, frame_cnt, 0])
                case 1: test.writerow([image_dir, 1, frame_cnt, 0])
                case 2: val.writerow([image_dir, 1, frame_cnt, 0])

    pass

if __name__=="__main__":
    tyro.cli(main)