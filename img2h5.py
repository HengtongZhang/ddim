import numpy as np
import h5py
import os

from PIL import Image
from tqdm import tqdm


target_size = 40
image_size = 64


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def process(f):
    with Image.open(f) as im:
        im = crop_center(im, crop_width=178, crop_height=178)
        im = im.resize((image_size, image_size), Image.LANCZOS)
        rtn = np.asarray(im, dtype=np.uint8)
        # im.save('11.jpg')
        return rtn


if __name__ == "__main__":
    ## Train
    prefix = "celeba/img_align_celeba/"
    l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))
    samples = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
    targets = np.zeros((len(l), target_size), dtype='uint8')
    fns = []
    targets_dict = {}

    with open("celeba/list_attr_celeba.txt") as fp:
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

    with h5py.File("celeba/celeba-{}.hdf5".format(image_size), 'w') as f:
        f['fns'] = fns
        f['samples'] = samples
        f['targets'] = targets
