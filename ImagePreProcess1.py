# image reading, convert, save with skimage
# Convert input image to small, normalized, grescale version
# J.Beale  12-Dec-2021

import os  # loop over images in directory
from skimage import exposure  # adaptive hist. equalization
import skimage.io  # to read & save images
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

# --------------------------------------------------------------

path = "C:\\Users\\beale\\Documents\\YOLO\\out\\car"  # read images from here
out_path = "C:\\Users\\beale\\Documents\\Umap\\raw"  # save them here


def inorm(img):
    target_dim = (15, 40)  # convert image to this size bitmap
    gray = rgb2gray(img)  # convert RGB image to greyscale
    img_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    img2 = resize(img_eq, target_dim, anti_aliasing=True)
    # imgplot = plt.imshow(img2)
    # plt.show()
    return(img_as_ubyte(img2))


for iname in os.listdir(path):
    if (iname.startswith("DH5_")) and (iname.lower().endswith(".jpg")):
        fname_in = os.path.join(path, iname)
        # print(fname_in)
        img = skimage.io.imread(fname=fname_in)  # color image input
        img_new = inorm(img)
        of_name = iname[0:-4] + ".png"
        fname_out = os.path.join(out_path, of_name)
        print(of_name)
        skimage.io.imsave(fname_out, img_new)  # save file

