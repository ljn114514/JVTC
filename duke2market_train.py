import os, torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.resnet import resnet50
from utils.dataset import imgdataset, imgdataset_camtrans
from utils.losses import Losses#, LocalLoss, GlobalLoss
from utils.evaluators import evaluate_all
from utils.lr_adjust import StepLrUpdater, SetLr

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
###########   HYPER   ###########
base_lr = 0.01
num_epoches = 100
batch_size = 128
K = 4
num_cam = 6
##########   DATASET   ###########
dataset_path = 'data/dataset/'
src_dir = dataset_path + 'DukeMTMC-reID/bounding_box_train/'
tar_dir = dataset_path + 'Market-1501/bounding_box_train_camstyle_merge/'
tar_dir_test = dataset_path + 'Market-1501/'

src_annfile = 'list_duke/list_duke_train.txt'
tar_annfile = 'list_market/list_market_train.txt'
tar_annfile_test = 'list_market/list_market_test.txt'

#resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth
imageNet_pretrain = 'resnet50-19c8e357.pth'


train_dataset = imgdataset(dataset_dir=src_dir, txt_path=src_annfile, transformer='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

train_dataset_t = imgdataset_camtrans(dataset_dir=tar_dir, txt_path=tar_annfile, 
	transformer='train', num_cam=num_cam, K=K)
train_loader_t = DataLoader(dataset=train_dataset_t, batch_size=int(batch_size/K), shuffle=True, num_workers=4, drop_last=True)

test_dataset_t = imgdataset(dataset_dir=tar_dir_test, txt_path=tar_annfile_test, transformer='test')
test_loader_t = DataLoader(dataset=test_dataset_t, batch_size=4, shuffle=False, num_workers=0)

###########   MODEL   ###########
model, param = resnet50(pretrained=imageNet_pretrain, num_classes=702)
model.cuda()
model = nn.DataParallel(model)#, device_ids=[0,1])

losses = Losses(K=K, 
	batch_size=batch_size, 
	bank_size=len(train_dataset_t), 
	ann_file=tar_annfile, 
	cam_num=num_cam)
losses = losses.cuda()
optimizer = torch.optim.SGD(param, lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

###########   TRAIN   ###########
target_iter = iter(train_loader_t)
for epoch in range(1, num_epoches+1):

	lr = StepLrUpdater(epoch, base_lr=base_lr, gamma=0.1, step=40)
	SetLr(lr, optimizer)

	print('-' * 10)
	print('Epoch [%d/%d], lr:%f'%(epoch, num_epoches, lr))

	running_loss_src = 0.0
	running_loss_local = 0.0
	running_loss_global = 0.0

	if (epoch)%5 == 0:
		losses.reset_multi_label(epoch)

	model.train()
	for i, source_data in enumerate(train_loader, 1):
		try:
			target_data = next(target_iter)
		except:
			target_iter = iter(train_loader_t)
			target_data = next(target_iter)

		image_src = source_data[0].cuda()
		label_src = source_data[1].cuda()
		image_tar = target_data[0].cuda()	
		image_tar = image_tar.view(-1, image_tar.size(2), image_tar.size(3), image_tar.size(4))
		label_tar = target_data[2].cuda()

		x_src = model(image_src)[0]
		x_tar = model(image_tar)[2]

		loss_all= losses(x_src, label_src, x_tar, label_tar, epoch)
		loss, loss_s, loss_l, loss_g = loss_all


		running_loss_src += loss_s.mean().item()
		running_loss_local += loss_l.mean().item()
		running_loss_global += loss_g.mean().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update_memory(x_tar, label_tar, epoch=epoch)

		if i % 50 == 0:
			print('  iter: %3d/%d,  loss src: %.3f, loss local: %.3f, loss global: %.3f'%(i, len(train_loader), running_loss_src/i, running_loss_local/i, running_loss_global/i))
	print('Finish {} epoch\n'.format(epoch))


print('evaluation..')
model.eval()
evaluate_all(model, test_loader_t, select_set='market')
		
if hasattr(model, 'module'):
	model = model.module
torch.save(model.state_dict(), 'snapshot/resnet50_duke2market_epoch%05d.pth'%(epoch))	
print('save snapshot:','snapshot/resnet50_duke2market_epoch%05d.pth'%(epoch))