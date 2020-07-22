import os, torch, random, cv2, math, glob
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from torch.nn import functional as F

class RandomErasing(object):
	def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
		self.EPSILON = EPSILON
		self.mean = mean

	def __call__(self, img):

		if random.uniform(0, 1) > self.EPSILON:
			return img

		for attempt in range(100):
			area = img.size()[1] * img.size()[2]

			target_area = random.uniform(0.02, 0.2) * area
			aspect_ratio = random.uniform(0.3, 3)

			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))

			if w <= img.size()[2] and h <= img.size()[1]:
				x1 = random.randint(0, img.size()[1] - h)
				y1 = random.randint(0, img.size()[2] - w)
				img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
				img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				return img

		return img

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = T.Compose([
	T.Resize((256,128)),
	T.RandomHorizontalFlip(),
	T.ToTensor(), 
	normalizer,
	RandomErasing(EPSILON=0.5) 
	])

test_transform = T.Compose([
	T.Resize((256,128)),
	T.ToTensor(),  
	normalizer  ])

class imgdataset(data.Dataset):
	def __init__(self, dataset_dir, txt_path, transformer = 'train'):

		self.transform = train_transform if transformer == 'train' else test_transform
		with open(txt_path) as f:
			line = f.readlines()
			self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
			self.label_list = [int(i.split()[1]) for i in line]
			self.cam_list = [int(i.split()[2]) for i in line]
			#self.cam_list = [int(i.split('c')[1][0]) for i in line]

	def __getitem__(self, index):
		im_path = self.img_list[index]
		image = Image.open(im_path)
		image = self.transform(image)				
		return image, self.label_list[index], self.cam_list[index]

	def __len__(self):
		return len(self.label_list)


class imgdataset_camtrans(data.Dataset):
	def __init__(self, dataset_dir, txt_path, transformer = 'train', num_cam=8, K=4):
		self.num_cam = num_cam
		self.transform = train_transform if transformer == 'train' else test_transform
		self.K = K
		with open(txt_path) as f:
			line = f.readlines()
			self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
			self.label_list = [int(i.split()[1]) for i in line]
			#self.cam_list = [int(i.split('c')[1][0]) for i in line]
			self.cam_list = [int(i.split()[2]) for i in line]

	def __getitem__(self, index):
		im_path = self.img_list[index]
		camid = self.cam_list[index]
		cams = torch.randperm(self.num_cam) + 1

		imgs = []
		for sel_cam in cams[0:self.K]:
			
			if sel_cam != camid:
				im_path_cam = im_path[:-4] + '_fake_' + str(camid) + 'to' + str(sel_cam.numpy()) + '.jpg'
			else:
				im_path_cam = im_path

			#print('im_path', camid, sel_cam,im_path_cam)
			image = Image.open(im_path_cam)
			image = self.transform(image)
			imgs.append(image.numpy())
			#imgs.append(image)

		imgs = np.array(imgs, np.float32)
		imgs = torch.from_numpy(imgs).float()

		return imgs, self.label_list[index], index

	def __len__(self):
		return len(self.label_list)