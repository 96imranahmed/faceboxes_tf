import cv2
import numpy as np
import tensorflow as tf

class DataService(object):
    def __init__(source_p, do_augment, data_path):
        self.source_p = source_p
        self.do_augment = do_augment
        self.data_path = data_path

    def read_image(self, loc):
        img = cv2.imread(loc)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def random_sample(self, count):
        choices = np.random.choice(self.source_p, size = count)
        imgs, boxes = [self.read_image(i['file_path']) for i in choices], [i['bbox'] for i in choices]
        if self.do_augment:
            imgs, boxes = self.augment(imgs, boxes)
        return imgs, boxes

    def augment(self, imgs, boxes):

        return imgs, boxes
    
