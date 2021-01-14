import os, sys
from collections import defaultdict

with open('list_market_train.txt') as f:
	lines = f.readlines()
	img_list = [i.split()[0] for i in lines]

#id_seqs = defaultdict(list)
seq_frames = defaultdict(list)
for item in img_list:
	info = item.split('_')
	label = info[0]
	seq_id = info[1]
	frame = info[2]
	#id_seqs[label].append(seq_id)
	seq_frames[seq_id].append(int(frame))

seq_frames_accum = defaultdict(list)
for key in seq_frames.keys():
	seq_idx = int(key[-1])

	if seq_idx == 1:
		seq_frames_accum[key] = 0#max(seq_frames[key])
	else:
		accum = 0
		for i in range(1, seq_idx):
			accum_key = key[0:3] + str(i)
			accum = accum + max(seq_frames[accum_key])
		seq_frames_accum[key] = accum

# keys = seq_frames_accum.keys()
# keys = list(keys)
# keys.sort()
# for key in keys:
# 	print(key, seq_frames_accum[key], max(seq_frames[key]))


with open('list_market_test.txt') as f:
	lines = f.readlines()

file = open('list_market_test_new.txt', 'w')
for line in lines:
	img_name = line.split()[0]
	#img_name = img_name.split('/')
	seq_id = img_name.split('/')[1].split('_')[1]

	line = line.split()
	frame_id = int(line[3])
	new_line = "%s %s %s %d\n"%(line[0], line[1], line[2], frame_id+seq_frames_accum[seq_id])
	file.write(new_line)