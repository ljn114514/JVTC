import os, torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from scipy.spatial.distance import cdist

from utils.util import cluster, get_info
from utils.util import extract_fea_camtrans, extract_fea_test
from utils.resnet import resnet50
from utils.dataset import imgdataset, imgdataset_camtrans
from utils.rerank import re_ranking
from utils.st_distribution import get_st_distribution
from utils.evaluate_joint_sim import evaluate_joint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_path = 'data/dataset/'
ann_file_train = 'list_market/list_market_train.txt'
ann_file_test = 'list_market/list_market_test.txt'

snapshot = 'snapshot/resnet50_duke2market_epoch00100.pth'

num_cam = 6
###########   DATASET   ###########
img_dir = dataset_path + 'Market-1501/bounding_box_train_camstyle_merge/'
train_dataset = imgdataset_camtrans(dataset_dir=img_dir, txt_path=ann_file_train, 
	transformer='test', K=num_cam, num_cam=num_cam)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=4)

img_dir = dataset_path + 'Market-1501/'
test_dataset = imgdataset(dataset_dir=img_dir, txt_path=ann_file_test, transformer='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

###########   TEST   ###########
model, _ = resnet50(pretrained=snapshot, num_classes=702)
model.cuda()
model.eval()

print('extract feature for training set')
train_feas = extract_fea_camtrans(model, train_loader)
_, cam_ids, frames = get_info(ann_file_train)

print('generate spatial-temporal distribution')
dist = cdist(train_feas, train_feas)
dist = np.power(dist,2)
#dist = re_ranking(original_dist=dist)
labels = cluster(dist)
num_ids = len(set(labels))
print('cluster id num:', num_ids)
distribution = get_st_distribution(cam_ids, labels, frames, id_num=num_ids, cam_num=num_cam)

print('extract feature for testing set')
test_feas = extract_fea_test(model, test_loader)

print('evaluation')
evaluate_joint(test_fea=test_feas, st_distribute=distribution, ann_file=ann_file_test, select_set='market')