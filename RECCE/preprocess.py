import os
import cv2
from retinaface import RetinaFace
# from argparse import ArgumentParser
import tyro
from tqdm import tqdm
from colorama import init, Fore
init(autoreset=True)
import pickle
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# /data4/FaceForensics++/
# ├── manipulated_sequences
# │   ├── DeepFakeDetection
# │   │   ├── c23
# │   │   │   └── videos
# │   │   ├── c40
# │   │   │   └── videos
# │   │   └── raw
# │   │       └── videos
# │   ├── Deepfakes
# │   │   ├── c23
# │   │   │   └── videos
# │   │   ├── c40
# │   │   │   └── videos
# │   │   └── raw
# │   │       └── videos
# │   ├── Face2Face
# │   │   ├── c23
# │   │   │   └── videos
# │   │   └── c40
# │   │       └── videos
# │   ├── FaceShifter
# │   │   ├── c23
# │   │   │   └── videos
# │   │   └── c40
# │   │       └── videos
# │   ├── FaceSwap
# │   │   ├── c23
# │   │   │   └── videos
# │   │   └── c40
# │   │       └── videos
# │   └── NeuralTextures
# │       ├── c23
# │       │   └── videos
# │       └── c40
# │           └── videos
# └── original_sequences
#     ├── actors
#     │   ├── c23
#     │   │   └── videos
#     │   ├── c40
#     │   │   └── videos
#     │   └── raw
#     │       └── videos
#     └── youtube
#         ├── c23
#         │   └── videos
#         ├── c40
#         │   └── videos
#         └── raw
#             └── videos

# def extract_frames(video_path, save_dir, target,):
#     # 打开视频文件
#     # video_path = '/data3/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4'
#     cap = cv2.VideoCapture(video_path)

#     frame_count = 0
#     images=[]
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 检测人脸
#         faces = RetinaFace.detect_faces(frame)

#         # 提取人脸并保存
#         for face in faces.values():
#             x1, y1, x2, y2 = face['facial_area']
#             face_image = frame[y1:y2, x1:x2]
#             idx=os.path.join(save_dir, f'{frame_count}.jpg')
#             cv2.imwrite(idx, face_image)
#             images.append((idx, target))
#         frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()
#     return images
#     pass

def extract_frames(video_path, save_dir, target):
    # 打开视频文件
    # video_path = '/data3/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    images=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_count % 3:
            frame_count += 1
            continue
        if not ret:
            break

        # 检测人脸
        faces = RetinaFace.detect_faces(frame)
        # exit()
        # 提取人脸并保存
        # print(faces.values())
        faces=sorted(list(faces.values()), key=lambda x : x['score'], reverse=True)
        # print(faces)
        # break
        # for face in faces.values():
        if len(faces)==0:
            print(video_path)
            continue
        face=faces[0]
        x1, y1, x2, y2 = face['facial_area']
        face_image = frame[y1:y2, x1:x2]

        idx=os.path.join(save_dir, f'{frame_count}.jpg')
        cv2.imwrite(idx, face_image)
        images.append((idx, target))
        frame_count += 1
        # exit()

    cap.release()
    cv2.destroyAllWindows()
    return images


config={
    'FF++':{
        'path':'/data4/FaceForensics++/',
        'save':'retinaface',
        'compression':'c40',
        'realfake':{
            'manipulated_sequences':'all', # or a list containing deepfake types
            'original_sequences':['youtube']
        },
        # 'fn':extract_ffpp
    }
}

def extract_ffpp():
    root=config['FF++']['path']
    todo=config['FF++']['realfake']
    save=config['FF++']['save']
    compression=config['FF++']['compression']
    _, dirs, _=next(os.walk(root))
    img=[]
    for d in dirs:
        # types=None
        # print(d)
        if d in todo.keys():
            subfolder=os.path.join(root, d)
            if todo[d] == 'all':
                types=os.listdir(subfolder)
            else:
                types=todo[d]
        else:
            continue
        for t in types:
            print(Fore.GREEN + d + ' ' + t)
            os.makedirs(os.path.join(root, save, d, t), exist_ok=True)
            vids=sorted(os.listdir(os.path.join(root, d, t, compression, 'videos')))
            vpar=tqdm(vids)
            for v in vpar:
                if not v.endswith('.mp4'):
                    continue
                fig_subdir=v.split('.')[0]
                vpar.set_description(v + str(len(img)))
                vid_path=os.path.join(root, d, t, compression, 'videos', v)
                save_dir=os.path.join(root, save, d, t, fig_subdir)
                os.makedirs(save_dir, exist_ok=True)
                # idx=os.path.join(save, d, t)
                if t == 'youtube' or t == 'actor':
                    target=0
                else:
                    target=1
                images=extract_frames(vid_path, save_dir, target)
                # print(images)
                # exit()
                img.extend(images)
    with open(os.path.join(root, save), 'wb+') as f:
        pickle.dump(f, img)
    pass

def extract(dataset:str='FF++'):
    if dataset == 'FF++':
        extract_ffpp()
    else:
        raise NotImplementedError
    pass

if __name__ == "__main__":
    tyro.cli(extract)