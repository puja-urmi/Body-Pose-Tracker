#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt

def show_keypoints(image, kp):
    plt.imshow(image)
    plt.scatter(kp[:, 0], kp[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()

