# -*- coding: utf-8 -*-
"""
UMAP plot for input images clustered according to similarity.

Created on Sun Dec 12 11:28:45 2021
@author: jbeale
https://umap-learn.readthedocs.io/en/latest/parameters.html

"""

import numpy as np
import seaborn as sns
import umap
import umap.plot
from math import sqrt
import os  # loop over images in directory
import skimage.io  # to read & save images
from skimage.util import img_as_float
import pandas as pd

# ---------------------------------------------------------------
#  cartesian distance between two points (a,b are 2-elem vectors)


def jdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dist = sqrt(dx*dx + dy*dy)
    return(dist)


# -------------------------------------------------------
#  find index and distance of the nearest elment in ar[] to point p


def jmin(p, ar):
    mdist = jdist(p, ar[0])
    nearest = 0
    for i in range(len(ar)):
        dist = jdist(p, ar[i])
        if (dist < mdist):
            mdist = dist
            nearest = i
    return(nearest, mdist)


# -----------------------------------------------

path = "C:\\Users\\beale\\Documents\\Umap\\raw2"  # raw input images here
img_count = 10000   # how many images to consider
px_count = 53*20    # how many pixels in each image (x * y)
dt = np.dtype(('U', 25))  # string with filename minus extension

sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
np.random.seed(17)
# data = np.random.rand(800, 768)  # 1st var: items  2nd var: parts of item
data = np.zeros((img_count, px_count), dtype=np.float32)  # items,parts of item
pname = np.ndarray(img_count, dt)

# Loop over (img_count) input image files and add them into data[]
# also add the filenames to pname[]
i = 0  # item index
for iname in os.listdir(path):
    if (iname.startswith("DH5_")) and (iname.lower().endswith(".png")):
        fname_in = os.path.join(path, iname)
        print(i, fname_in)
        pname[i] = (iname[0:25])
        img = img_as_float(skimage.io.imread(fname=fname_in))  # img input
        data[i, :] = img.reshape(px_count)
        i = i + 1
        if (i >= img_count):
            break

# --- Generate the clustered similarity map for the data
# mapper = umap.UMAP(n_neighbors=30, min_dist=0.05).fit(data)
mapper = umap.UMAP(densmap=True, dens_lambda=1,
                   n_neighbors=10, min_dist=0.1).fit(data)
# mapper = umap.UMAP(n_neighbors=10, min_dist=0).fit(data)
umap.plot.diagnostic(mapper, diagnostic_type='vq')

umap.plot.points(mapper)
umap.plot.connectivity(mapper, show_points=True)  # inter-connection lines
umap.plot.diagnostic(mapper, diagnostic_type='pca')

local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')

# display plot on an interactive web page
hover_data = pd.DataFrame({'index': np.arange(img_count),
                           'label': pname[:img_count]})
p = umap.plot.interactive(mapper, labels=pname[:img_count],
                          hover_data=hover_data, point_size=3)
umap.plot.show(p)

"""
i_cent = 8888   # index of item in center of ROI
i_edge = 8612   # index of item at edge of ROI
d_center = mapper.embedding_[i_cent]  # coords of data at center ROI
r_active = jdist(d_center, mapper.embedding_[i_edge])  # radius of ROI

j = 0
for i in range(len(mapper.embedding_)):
    r_test = jdist(d_center, mapper.embedding_[i])
    if (r_test < r_active):
        print(j, i, r_test, pname[i])
        j += 1

n_ROI = j  # count of data items within ROI
data1 = np.zeros((n_ROI, px_count), dtype=np.float32)  # ROI only
# imap1 = np.zeros(n_ROI, dtype=np.int32)  # map new index to original index
pname1 = np.ndarray(n_ROI, dt)  # filenames in ROI
j = 0
for i in range(len(mapper.embedding_)):
    r_test = jdist(d_center, mapper.embedding_[i])
    if (r_test < r_active):
        data1[j, :] = data[i, :]
        pname1[j] = pname[i]  # save the ROI name from original names
        j += 1

# mapper1 = umap.UMAP(n_neighbors=10, min_dist=0.05).fit(data1)
# mapper1 = umap.UMAP(n_neighbors=10, min_dist=0.02).fit(data1)
mapper1 = umap.UMAP(densmap=True, dens_lambda=4,
                    n_neighbors=10, min_dist=0.2).fit(data1)

umap.plot.diagnostic(mapper1, diagnostic_type='vq')

"""

"""
# reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
# embedding = reducer.fit_transform(data)
# import joblib
# save_filename = 'Car25k-reduc1.sav'
# joblib.dump(reducer, save_filename)
# loaded_reducer = joblib.load(save_filename)


"""

"""
hover_data1 = pd.DataFrame({'index': np.arange(n_ROI),
                           'label': pname1})
p1 = umap.plot.interactive(mapper1, labels=pname1,
                           hover_data=hover_data1, point_size=3)
umap.plot.show(p1)
"""

"""
# fit = umap.UMAP(n_neighbors=60, min_dist = 0.15)
# u = fit.fit_transform(data)
# plt.scatter(u[:,0], u[:,1], c=data[:,0:4])
# plt.scatter(u[:, 0], u[:, 1])
# plt.title('UMAP embedding of car images')
"""


"""
reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
embedding = reducer.fit_transform(data)
import joblib
save_filename = 'CarB10kB-reduce.sav'
joblib.dump(reducer, save_filename)


fout = "G2-cars10k.txt"

with open(fout, 'w') as f:
    f.write("idx,x,y,fname\n")
    for j in range(len(data)):
        f.write("%d, %5.3f,%5.3f, %s.jpg\n" % (j, mapper.embedding_[j, 0],
                                        mapper.embedding_[j,1], pname[j]))
f.close()

fout = "G2-cars3636.csv"
with open(fout, 'w') as f:
    f.write("idx,x,y,fname\n")
    for j in range(n_ROI):
        f.write("%d, %5.3f,%5.3f, %s.jpg\n" % (j, mapper1.embedding_[j, 0],
                                        mapper1.embedding_[j,1], pname1[j]))
f.close()

"""


"""
# find data element nearest each to each regularly-spaced grid vertex

p = np.array([1, 1])

xsteps = 10
ysteps = 20
xa = -3    # min..max X range on UMAP axis
xb = 12
ya = -4    # min..max Y range on UMAP axis
yb = 14
xs = (xb - xa) / xsteps
ys = (yb - ya) / ysteps

print("x,y,n,dist,fname")
for x in range(xsteps):
    for y in range(ysteps):
        xp = xa + x*xs
        yp = ya + y*ys
        p = np.array([xp, yp])
        (index, dist) = jmin(p, mapper1.embedding_)
        print("%d,%d,%d,%5.2f,%s" %
              (x, y, index, dist, pname1[index]), end='\n')
"""
