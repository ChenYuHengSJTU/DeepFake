from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor, resized_crop
import cv2

def init_ff(dataset='all',phase='test', compression='c23'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path=f'data/FaceForensics++/original_sequences/youtube/{compression}/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/{compression}/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list



def init_dfd(compression='c23'):
	real_path=f'data/FaceForensics++/original_sequences/actors/{compression}/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path=f'data/FaceForensics++/manipulated_sequences/DeepFakeDetection/{compression}/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list


def init_dfdcp(phase='test'):

	phase_integrated={'train':'train','val':'train','test':'test'}

	all_img_list=[]
	all_label_list=[]

	with open('data/DFDCP/dataset.json') as f:
		df=json.load(f)
	fol_lab_list_all=[[f"data/DFDCP/{k.split('/')[0]}/videos/{k.split('/')[-1]}",df[k]['label']=='fake'] for k in df if df[k]['set']==phase_integrated[phase]]
	name2lab={os.path.basename(fol_lab_list_all[i][0]):fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all=[f[0] for f in fol_lab_list_all]
	fol_list_all=[os.path.basename(p)for p in fol_list_all]
	folder_list=glob('data/DFDCP/method_*/videos/*/*/*.mp4')+glob('data/DFDCP/original_videos/videos/*/*.mp4')
	folder_list=[p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list=[name2lab[os.path.basename(p)] for p in folder_list]
	

	return folder_list,label_list




def init_ffiw():
	# assert dataset in ['real','fake']
	path='data/FFIW/FFIW10K-v1-release/'
	folder_list=sorted(glob(path+'source/val/videos/*.mp4'))+sorted(glob(path+'target/val/videos/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list



def init_cdf():

	image_list=[]
	label_list=[]

	video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
	with open(video_list_txt) as f:
		
		folder_list=[]
		for data in f:
			# print(data)
			line=data.split()
			# print(line)
			path=line[1].split('/')
			folder_list+=['data/Celeb-DF-v2/'+path[0]+'/videos/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list


class SimpleDataset(Dataset):
	def __init__(self, data, label):
		# super().__init__(SimpleDataset)
		self.data=data
		self.length=len(data)
		self.label=label
		assert len(data) == len(label), "LENGTH OF DATA AND LABEL MUST BE A MATCH"
	
	def __getitem__(self, idx):
		# with open(self.data[idx], 'r') as f:
			# image=Image.open(f)
		image=Image.open(self.data[idx])
		if self.label[idx] == 0:
			image=image.crop((150, 0, 1898, 1748)).resize((540, 540))
		else:
			image=image.crop((210, 0, 750, 540))
		return to_tensor(image), self.label[idx]

	def __len__(self):
		return self.length


def init_avatar():
	image_list=[]
	label_list=[]

	# real is 0;any other folder name will be fake, namely 1
	image_list_root='/Code/SelfBlendedImages/data/Avatar'
	subdir=sorted(os.listdir(image_list_root))
	spar=tqdm(subdir)
	for subd in spar:
		if subd == 'real':
			target=1
		else:
			target=0
		subdd=os.path.join(image_list_root, subd)
		spar.set_description(subdd)
		spar.write(str(target))
		images=sorted(list(glob(f'{subdd}/*/*g')))
		label_list.extend([target]*len(images))
		image_list.extend(images)
	print(len(image_list), len(label_list))
	return SimpleDataset(image_list, label_list)


# def init_talking():
# 	label_list=[]
# 	folder_list=[]
# 	video_root='/Code/SelfBlendedImages/data/TalkingHead/'
# 	videos=sorted(os.listdir(video_root))
# 	for v in videos:
# 		# if v.endswith('mov'):
# 			# continue
# 		if os.path.isdir(os.path.join(video_root, v)):
# 			continue
# 		folder_list+=['data/TalkingHead/'+v]
# 		if 'render' in v:
# 			label_list+=[1]
# 		else:
# 			label_list+=[0]
# 	return folder_list,label_list

class TalkingHeadDataset(Dataset):
	def __init__(self, image, label):
		self.data=image
		self.label=label
		self.length=len(image)
		assert len(image) == len(label),f"IMAGE {len(image)} AND LABLE {len(label)} NUMBER MUST MATCH EACH OTHER"

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return self.length

def init_talking():
	label_list=[]
	folder_list=[]
	frame_list=[]
	video_root='/Code/SelfBlendedImages/data/TalkingHead/'
	videos=sorted(glob(f'{video_root}/*/*'))
	vpar=tqdm(videos)
	for v in vpar:
		if 'real' in v:
			target=0
		else:
			target=1
		vpar.set_description(v)
		vpar.write(str(target))
		cap=cv2.VideoCapture(v)
		vpar.write(str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		vpar.write(str(len(frame_list)))
		vpar.write(str(len(label_list)))
		vpar.write('-'*30)
		# label_list.extend([target]*int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		frame_count=0
		while True:
			ret, frame=cap.read()
			if not ret:
				break
			frame_count+=1
			frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# vpar.write(str(type(frame)))
			
			# print(frame.size)
			h,w,c=frame.shape
			# frame=frame.permute(2,0,1)
			frame=frame.reshape(c,h,w)
			frame_list.append(frame)
			# print(frame.shape)
		cap.release()
		label_list.extend([target]*frame_count)
	return TalkingHeadDataset(frame_list,label_list)