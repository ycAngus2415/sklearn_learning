#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:00:48 2017

@author: yangchao
"""
from sklearn import datasets as d
from matplotlib import pyplot as plt
import numpy as np
blobs = d.make_blobs(200)
dd = d.load_boston()

plt.figure()
colors = np.array(['r', 'g' , 'b'])
plt.scatter(blobs[0][:, 0], blobs[0][:,1], color=colors[blobs[1].astype(int)])