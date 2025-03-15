import pickle

with open('/data3/FaceForensics++/retinaface/train_c40.pickle', 'rb+') as f:
    train=pickle.load(f)
    print(train)