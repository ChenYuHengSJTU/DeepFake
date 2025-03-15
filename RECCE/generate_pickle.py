import os
import pickle
import colorama
from colorama import init, Fore
init(autoreset=True)
from tqdm import tqdm
from random import shuffle

train_ratio=0.7
val_ratio=0.1
test_ratio=0.2



def dump_pickle(source:str='/data3/FaceForensics++/retinaface'):
    all_record=[]   
    # base_name=os.path.basename(source)
    # extraction_path=os.path.join(base_name, 'extraction')
    # os.makedirs(extraction_path, exist_ok=True)
    fake_source=os.path.join(source, 'manipulated_sequences')
    real_source=os.path.join(source, 'original_sequences', 'youtube')
    fake_type=sorted(os.listdir(fake_source))
    print(fake_type)
    for ft in fake_type:
        ft_dir=os.path.join(fake_source, ft)
        relative_path=os.path.join('manipulated_sequences', ft)
        print(Fore.GREEN + ft_dir)
        ipar=tqdm(sorted(os.listdir(ft_dir)))
        for imgdir in ipar:
            ipar.set_description(imgdir)
            imgs=sorted(os.listdir(os.path.join(fake_source, ft, imgdir)))
            for i in imgs:
                all_record.append((os.path.join(relative_path, imgdir, i), 1))
    rpar=tqdm(sorted(os.listdir(real_source)))
    for real in rpar:
        rpar.set_description(real)
        for img in sorted(os.listdir(os.path.join(real_source, real))):
            all_record.append((os.path.join('original_sequences', 'youtube', real, img), 0))
    # print(len(all_record))    
    # print(all_record)
    shuffle(all_record)
    print(type(all_record))
    sz=len(all_record)
    train_sz=int(sz*train_ratio)
    val_sz=int(sz*val_ratio)
    train=all_record[:train_sz]
    val=all_record[train_sz:train_sz+val_sz]
    test=all_record[train_sz+val_sz:]
    with open(os.path.join(source, 'train_c40.pickle'), 'wb+') as f:
        pickle.dump(train, f)
    with open(os.path.join(source, 'val_c40.pickle'), 'wb+') as f:
        pickle.dump(val, f)
    with open(os.path.join(source, 'test_c40.pickle'), 'wb+') as f:
        pickle.dump(test, f)


if __name__ == '__main__':
    dump_pickle()