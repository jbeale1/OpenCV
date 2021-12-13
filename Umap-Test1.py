# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:28:45 2021

@author: beale
https://umap-learn.readthedocs.io/en/latest/parameters.html

"""

import numpy as np
import seaborn as sns
import umap
import umap.plot

import os  # loop over images in directory
import skimage.io  # to read & save images
from skimage.util import img_as_float

# ---------------------------------------------------------------

path = "C:\\Users\\beale\\Documents\\Umap\\raw"  # get input from here
# img_count = 10000    # how many images to consider
img_count = 8790    # how many images to consider

px_count = 40*15    # how many pixels in each image (x * y)

sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
np.random.seed(42)
# data = np.random.rand(800, 768)  # 1st var: items  2nd var: parts of item
data = np.zeros((img_count, px_count), dtype=np.float32)  # items,parts of item

i = 0  # item index
for iname in os.listdir(path):
    if (iname.startswith("DH5_")) and (iname.lower().endswith(".png")):
        fname_in = os.path.join(path, iname)
        print(i, fname_in)
        img = img_as_float(skimage.io.imread(fname=fname_in))  # img input
        data[i, :] = img.reshape(px_count)
        i = i + 1
        if (i >= img_count):
            break


mapper = umap.UMAP(n_neighbors=30, min_dist=0.10).fit(data)

# reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
# embedding = reducer.fit_transform(data)
# import joblib
# save_filename = 'Car25k-reduc1.sav'
# joblib.dump(reducer, save_filename)
# loaded_reducer = joblib.load(save_filename)


umap.plot.points(mapper)
umap.plot.connectivity(mapper, show_points=True)  # inter-connection lines
umap.plot.diagnostic(mapper, diagnostic_type='pca')
umap.plot.diagnostic(mapper, diagnostic_type='vq')
local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')

# fit = umap.UMAP(n_neighbors=60, min_dist = 0.15)
# u = fit.fit_transform(data)
# plt.scatter(u[:,0], u[:,1], c=data[:,0:4])
# plt.scatter(u[:, 0], u[:, 1])
# plt.title('UMAP embedding of car images')

"""
reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
embedding = reducer.fit_transform(data)
import joblib
save_filename = 'Car10kC-reduce.sav'
joblib.dump(reducer, save_filename)
"""
