#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from shutil import copyfile


# In[2]:


def scale_annotations(path, copypath):
    # make a copy of the annotations file
    # copyfile("insert path to keypoint annotations", "insert the same file path just with a _copy appended")
    copyfile(path, copypath)

    # now we will modify the annotations in the copied file
    f = open(copypath, 'r')
    data_keypoints = json.load(f)
    f.close()

    for data in data_keypoints['images']:
        data['height'] //= 5
        data['width'] //= 5

    for data in data_keypoints['annotations']:
        kpts = np.array(data['keypoints'])
        kpts = np.reshape(kpts, (17,3))
        kpts[:, 0] //= 5
        kpts[:, 1] //= 5
        kpts[:, 2] //= 1
        data['keypoints'] = kpts.reshape((51,)).tolist()
        data['bbox'] = [coordinate//5 for coordinate in data['bbox']]
        data['area'] //= 25



    f = open(copypath, "w")
    json.dump(data_keypoints, f)
    f.close()

