import sys
import pickle
import os
import cv2
import numpy as np

PATH_TO_DATA = '../WIDER/train_images/'
PATH_TO_ANNS = '../WIDER/wider_face_train_bbx_gt.txt'

# PATH_TO_DATA = '../WIDER/test_images/'
# PATH_TO_ANNS = '../WIDER/wider_face_val_bbx_gt.txt'

anns_file = open(PATH_TO_ANNS, "r")
anns_file = tuple(anns_file)
data = []
stage = 0
cur_entry = {}
cur_count = 0
i = 0
for line in anns_file:
    print('Processing: ', i, ' out of ', len(anns_file), '            ', end = '\r')
    i += 1
    if stage == 0:
        cur_entry['file_path'] = line
        img = cv2.imread(PATH_TO_DATA + line.strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cur_entry['img'] = img
        stage = (stage+1)%3
        cur_entry['hwc'] = img.shape[0:3]
    elif stage == 1: 
        cur_entry['n'] = int(line.strip())
        cur_count = 0
        stage = (stage+1)%3
    elif stage == 2:
        try:
            box = [int(i) for i in line.strip().split()[0:4]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            if box[0] > box[2]:
                raise ValueError("Something wrong", box, cur_entry)
            if 'bbox' in cur_entry:
                cur_entry['bbox'].append(box)
            else:
                cur_entry['bbox'] = [box]
        except: 
            print('Something went wrong with ', line.strip().split())
        cur_count += 1
        if cur_count == cur_entry['n']:
            stage = (stage+1)%3
            cur_entry['bbox'] = np.array(cur_entry['bbox'])
            data.append(cur_entry)
            cur_entry = {}

pickle.dump(obj = data, file = open('./wider_train.p', 'wb'))
# pickle.dump(obj = data, file = open('./wider_test.p', 'wb'))

# The following files contain invalid bboxes (remove negatives): 
# 54--Rescue/54_Rescue_rescuepeople_54_29.jpg
# 7--Cheering/7_Cheering_Cheering_7_17.jpg