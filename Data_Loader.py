#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import cv2
import imutils
#--------------------------------------------------------------------------------------------------------------------------


# In[2]:


class Pose_Dataset(Dataset):
    def __init__(self, annotations_path, img_path, transform = None):
        f = open(annotations_path, 'r')
        self.all_annotations = json.load(f)
        self.keypoints = self.all_annotations['annotations']
        f.close()
        self.annnotations_path = annotations_path
        self.img_path = img_path
        self.transform = transform
    def __len__(self):
        return len(self.keypoints)
    def __getitem__(self, idx):
        def form_gaussian_batch(sigma, kpts, img):
            batch = np.zeros((img.shape[0], img.shape[1], 17))
            for i in range(0,17):
                batch[:, :, i] = get_gaussian(img.shape, kpts[i, 0], kpts[i, 1], sigma)
            return batch
    
        def get_gaussian(output_shape, x, y, sigma):
            xx, yy= np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0]))
            return ((1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-(yy-y)**2/(2*sigma**2)))*((1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-(xx-x)**2/(2*sigma**2)))
        
        if torch.is_tensor(idx):
            idx = idx.to_list()
        anns = self.keypoints[idx]
        image = cv2.imread(os.path.join(self.img_path, anns['image_name']))
        image = cv2.resize(image, (0,0), fx = 0.2, fy = 0.2)
        bbox = anns['bbox']
        cropped_img = image[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
        
        kpts = np.array(anns['keypoints'], dtype = np.int32)
        kpts = np.reshape(kpts, (17,3))
        kpts[:, 0] = kpts[:, 0] - int(bbox[0])
        kpts[:, 1] = kpts[:, 1] - int(bbox[1])
        # now generate gaussian maps as target
        sigmas = [13, 6.5, 3, 1.5]
        target = []
        for sigma in sigmas:
            target.append(form_gaussian_batch(sigma, kpts, cropped_img))
        sample = {'image' : cropped_img, 'target' : target}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
#--------------------------------------------------------------------------------------------------------------------------


# In[3]:


def my_collate(Batch):
    image = []
    targets = []
    for item in Batch:
        img = item['image']
        if(img.shape[0] * img.shape[1] < 10000):
            continue
        image.append(img)
        targets.append(item['target'])
    return {'image' : image , 'target' : targets}
#--------------------------------------------------------------------------------------------------------------------------


# In[4]:


class RandomFlip(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if(np.random.random_sample() > self.p):
            flipped_image = cv2.flip(image, 1)
            flipped_target = []
            for batch in target:
                flipped_batch = np.zeros(batch.shape)
                for i in range(0, 17):
                    flipped_batch[:, :, i] = cv2.flip(batch[:, :, i], 1)
                corrected_batch = np.zeros(batch.shape)
                corrected_batch[:, :, 0] = flipped_batch[:, :, 0]
                for j in range(1, 9):
                    corrected_batch[:, :, ((2*j) -1)] = flipped_batch[:, :, (2*j)]
                    corrected_batch[:, :, (2*j)] = flipped_batch[:, :, ((2*j) -1)]
                flipped_target.append(corrected_batch)
            return {'image' : flipped_image, 'target' : flipped_target}
        return sample
#--------------------------------------------------------------------------------------------------------------------------


# In[13]:


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        angle = np.random.random_sample()*self.degree
        angle = angle if (np.random.random_sample() > 0.5) else (-1 * angle)
        rotated_image = imutils.rotate_bound(np.float32(image)/255, angle = angle)
        rotated_target = []
        for batch in target:
            h, w = imutils.rotate_bound(np.float32(batch[:, :, 0]), angle = angle).shape
            rotated_batch = np.zeros((h, w, 17))
            for i in range(0, 17):
                rotated_batch[:, :, i] = imutils.rotate_bound(np.float32(batch[:, :, i]), angle = angle)
            rotated_target.append(rotated_batch)
        return {'image' : rotated_image, 'target' : rotated_target}
#--------------------------------------------------------------------------------------------------------------------------


# In[14]:


class ToTensor(object):
    def __call__(self, sample):
        image, targets = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        tensor_targets = []
        for batch in targets:
            tensor_targets.append(torch.from_numpy(batch)) 
        return {'image': image,
                'target': tensor_targets}


# In[17]:


if __name__ == '__main__':
    
    pose_keypoints = Pose_Dataset('../annotations/vru_keypoints_val_copy.json', '../images/val')
    train_loader = DataLoader(pose_keypoints, batch_size = 8, shuffle = False, num_workers = 0,collate_fn = my_collate
                             ,pin_memory = True)
    #--------------------------------------------------------------------------------------------------------------------------

    batch_data = next(iter(train_loader))
    #--------------------------------------------------------------------------------------------------------------------------

    img = batch_data['image'][1]
    plt.imshow(img[:, :, [2,1,0]])
    anno = batch_data['target'][1][3][:, :, 16]
    plt.imshow(anno, alpha = 0.5)
    #--------------------------------------------------------------------------------------------------------------------------

    tsfm = RandomRotate(degree = 30)
    transformed_sample = tsfm(pose_keypoints[1])
    #--------------------------------------------------------------------------------------------------------------------------

    plt.imshow(transformed_sample['image'][:, :, [2, 1, 0]])
    plt.imshow(transformed_sample['target'][2][:, :, 11], alpha = 0.5)


# In[ ]:




