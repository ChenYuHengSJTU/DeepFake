import lmdb
import os
from tqdm import tqdm
from colorama import init, Fore
import tyro
import cv2
from random import random
import pickle

init(autoreset=True)

def get_split():
    rand_val=random()
    if rand_val<0.7:
        return 0
    elif rand_val<0.8:
        return 1
    else:
        return 2

def main(src_folder:str='/data3/FaceForensics++', compression:str='c23'):
    os.makedirs(os.path.join(src_folder, 'retinaface', 'lmdb'), exist_ok=True)
    TB=1099511627776
    train=lmdb.open(os.path.join(src_folder, 'retinaface', 'lmdb','train_img'), map_size=TB*2)
    test=lmdb.open(os.path.join(src_folder, 'retinaface', 'lmdb', 'test_img'), map_size=TB*2)
    val=lmdb.open(os.path.join(src_folder, 'retinaface', 'lmdb', 'val_img'), map_size=TB*2)
    train_cnt, test_cnt, val_cnt=0,0,0
    # train_label=lmdb.open(os.path.join(src_folder, 'lmdb', 'train_label'))
    # test_label=lmdb.open(os.path.join(src_folder, 'lmdb', 'train_label'))
    # val_label=lmdb.open(os.path.join(src_folder, 'lmdb', 'val_lmdb'))
    fake_src=os.path.join(src_folder, 'retinaface', 'manipulated_sequences')
    real_src=os.path.join(src_folder, 'retinaface', 'original_sequences', 'youtube', compression)
    fake_types=sorted(os.listdir(fake_src))
    with train.begin(write=True) as train, test.begin(write=True) as test, val.begin(write=True) as val:
        # train=csv.writer(f0)
        # test=csv.writer(f1)
        # val=csv.writer(f2)
        for fk in tqdm(fake_types):
            if fk=='DeepFakeDetection':
                continue
            fake_videos_path=os.path.join(fake_src, fk, compression)
            print(Fore.GREEN+fk)
            vpar=tqdm(sorted(os.listdir(fake_videos_path)))
            for v in vpar:
                vpar.set_description(v)
                img_path=os.path.join(fake_videos_path, v)
                # frame_cnt, image_dir=get_frame_count(vid_path)
                for im in sorted(os.listdir(img_path)):
                    img=cv2.imread(os.path.join(img_path, im))
                    if img is None:
                        tqdm.write(Fore.RED+f'read {os.path.join(img_path, im)} error')
                        continue
                    split=get_split()
                    # vid_path=vid_path.split('.')[0]
                    match split:
                        case 0: 
                            train.put(str(train_cnt).encode(), pickle.dumps((img, 1)))
                            train_cnt+=1
                        case 1: 
                            test.put(str(test_cnt).encode(), pickle.dumps((img, 1)))
                            test_cnt+=1
                        case 2: 
                            val.put(str(val_cnt).encode(), pickle.dumps((img, 1)))
                            val_cnt+=1

        vpar=tqdm(sorted(os.listdir(real_src)))
        print(Fore.GREEN+'Real')
        for real in vpar:
            vpar.set_description(real)
            img_path=os.path.join(real_src, real)
            for im in sorted(os.listdir(img_path)):
                img=cv2.imread(os.path.join(img_path, im))
                if img is None:
                    tqdm.write(Fore.RED+f'read {os.path.join(img_path, im)} error')
                    continue
                split=get_split()
                match split:
                    case 0: 
                        train.put(str(train_cnt).encode(), pickle.dumps((img, 0)))
                        train_cnt+=1
                    case 1: 
                        test.put(str(test_cnt).encode(), pickle.dumps((img, 0)))
                        test_cnt+=1
                    case 2:
                        val.put(str(val_cnt).encode(), pickle.dumps((img, 0)))
                        val_cnt+=1

        # train.commit()
        # test.commit()
        # val.commit()
    pass

if __name__=="__main__":
    tyro.cli(main)