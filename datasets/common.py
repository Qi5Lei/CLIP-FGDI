# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import sys
import time
# sys.path.append('/home/zhaohuazhong/DGReID/')
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
from reidutils.file_io import PathManager
from collections import defaultdict

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.
    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        # handle grayscale mixed in RGB images
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True,last_id=0):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.last_id = last_id

        pid_set = set()
        pids = []
        cam_set = set()
        domains = set()
        domain = []
        dcmain = defaultdict(set)
        # self.p2d = defaultdict(set)
        for i in img_items:
            pid_set.add(i[1])
            pids.append(i[1])
            cam_set.add(i[2])
            domains.add(i[3])
            domain.append(i[3])
            dcmain[i[3]].add(i[2])
        p2d = dict(zip(pids,domain))
        domains = list(domains)
        domains.sort()

        self.demains = {}

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.domains = {list(domains)[i]: i for i in range(len(list(domains)))}

        self.p2d = {}

        print(self.domains)
        if relabel:
            self.pid_dict = dict([(p, i+self.last_id) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            for p, d in p2d.items():
                self.p2d[self.pid_dict[p]] = self.domains[d]
        else:
            for p, d in p2d.items():
                self.p2d[p] = self.domains[d]
        for elm in dcmain.keys():
            self.demains[elm] = dict([(p, i) for i, p in enumerate(dcmain[elm])])





    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        domain = img_item[3]
        img = read_image(img_path)

        # pdb.set_trace()
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid_agg = self.pid_dict[pid]

            rescamid = self.cam_dict[camid]
            # pid_expert = int(pid.split('_')[-1])
            return img, pid_agg, rescamid, 1, img_path, self.domains[domain],self.demains[domain][camid]
        else:
            return img, pid, camid, 1, img_path, self.domains[domain],self.demains[domain][camid]

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        trackid = 1
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


if __name__ == "__main__":
    read_image('/data/zhz_dataset/ReID/cuhk03/images_labeled/1_001_2_08.png')
