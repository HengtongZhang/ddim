import numpy as np
import h5py
import os

import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# from scipy.misc import imresize
target_size = 40
image_size = 64
num_cpus = multiprocessing.cpu_count()


def process(f):
    global image_size
    im = Image.open(f)
    im = im.resize((image_size, image_size), Image.LANCZOS)
    # im = imresize(im, (image_size, image_size), interp='bicubic')
    return im


if __name__ == "__main__":
    ## Train
    prefix = './img_align_celeba/'
    l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))
    samples = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
    targets = np.zeros((len(l), target_size), dtype='uint8')
    fns = []
    targets_dict = {}

    with open("list_attr_celeba.txt") as fp:
        lines = fp.readlines()
        for line in lines[2: ]:
            entries = line.split()
            targets_dict[entries[0]] = np.asarray(entries[1:], dtype=np.int8)

    for i in tqdm(range(len(l))):
        fn_path = l[i]
        samples[i] = process(fn_path)
        fn = fn_path.split('/')[-1]
        targets[i] = targets_dict[fn]
        fns.append(fn)

    with h5py.File('./celeba-64.hdf5', 'w') as f:
        f['fns'] = fns
        f['samples'] = samples
        f['targets'] = targets
