import os
from colorama import init, Fore
from tqdm import tqdm
import tyro
import re

def main(src:str='/Code/Deepfake/TALL4Deepfake'):
    with open(os.path.join(src, 'train.txt'), 'r+') as f1, open('train_fake.txt', 'w+') as f2, open('train_real.txt', 'w+') as f3:
        for l in tqdm(f1.readlines()):
            # path, start, end, label
            raw = re.sub(r"\s", "", l).split(",")
            if int(raw[-1]):
                f2.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
            else:
                f3.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
    with open(os.path.join(src, 'test.txt'), 'r+') as f1, open('test_fake.txt', 'w+') as f2, open('test_real.txt', 'w+') as f3:
        for l in tqdm(f1.readlines()):
            # path, start, end, label
            raw = re.sub(r"\s", "", l).split(",")
            if int(raw[-1]):
                f2.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
            else:
                f3.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
    with open(os.path.join(src, 'val.txt'), 'r+') as f1, open('val_fake.txt', 'w+') as f2, open('val_real.txt', 'w+') as f3:
        for l in tqdm(f1.readlines()):
            # path, start, end, label
            raw = re.sub(r"\s", "", l).split(",")
            if int(raw[-1]):
                f2.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
            else:
                f3.write(f'{raw[-1]},{raw[0]},{raw[2]}\n')
    pass

if __name__=='__main__':
    tyro.cli(main)