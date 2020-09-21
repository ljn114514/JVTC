import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch.nn import functional as F

def l2_dist(fea_query, fea_gallery):
	dist = np.zeros((fea_query.shape[0], fea_gallery.shape[0]), dtype = np.float64)
	for i in range(fea_query.shape[0]):
		dist[i, :] = np.sum((fea_gallery-fea_query[i,:])**2, axis=1)
	return dist
	

def cluster(dist, rho=1.6e-3):

	tri_mat = np.triu(dist, 1)
	tri_mat = tri_mat[np.nonzero(tri_mat)]
	tri_mat = np.sort(tri_mat,axis=None)
	top_num = np.round(rho*tri_mat.size).astype(int)
	eps = tri_mat[:top_num].mean()
	#low eps for training without source domain
	#eps = eps*0.8 
	#print('eps in cluster: {:.3f}'.format(eps))
	cluster = DBSCAN(eps=eps,min_samples=1,metric='precomputed', n_jobs=8)
	labels = cluster.fit_predict(dist)

	return labels


def get_info(file_path):
	with open(file_path) as f:
		lines = f.readlines()
		#self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in lines]
		labels = [int(i.split()[1]) for i in lines]
		cam_ids = [int(i.split()[2]) for i in lines]
		frames = [int(i.split()[3]) for i in lines]

	return labels, cam_ids, frames

def extract_fea_camtrans(model, loader):
	feas = []
	for i, data in enumerate(loader, 1):
		#break
		with torch.no_grad():
			image = data[0].cuda()

			batch_size = image.size(0)
			K = image.size(1)

			image = image.view(image.size(0)*image.size(1), image.size(2), image.size(3), image.size(4))
			#image = Variable(image).cuda()
			out = model(image)
			fea = out[2]
			fea = fea.view(batch_size, K, -1)
			fea = fea.mean(dim=1)
			fea = F.normalize(fea)
			feas.append(fea)

	feas = torch.cat(feas)
	#print('duke_train_feas', feas.size())
	return feas.cpu().numpy()

def extract_fea_test(model, loader):
	feas = []
	for i, data in enumerate(loader, 1):
		#break
		with torch.no_grad():
			image = data[0].cuda()
			out = model(image)
			fea = out[1]
			feas.append(fea)

	feas = torch.cat(feas)
	#print('duke_train_feas', feas.size())
	return feas.cpu().numpy()
