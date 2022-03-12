import multiprocessing
import os

import torch

from datadings.reader import MsgpackReader
from PIL import Image
from simplejpeg import decode_jpeg 

class ImagenetDatadings(torch.utils.data.Dataset):
    """Wrapper to Imagenet Datadings using map-style datasets"""

    def __init__(self, filename, transform=None):
        super(ImagenetDatadings, self).__init__()
        assert os.path.exists(filename)
        self.path = filename
        self.lock = multiprocessing.Lock()
        # buffer set to 512 KB
        self.__len = len(MsgpackReader(filename, buffering=5 * 1024))
        self.__reader = None
        self.transform = transform

    @property
    def reader(self):
       if self.__reader is None:
         self.__reader = MsgpackReader(self.path, buffering=5 * 1024)
       return self.__reader


    def __getitem__(self, index):
        reader = self.reader
        reader.seek(index)        
        sample = reader.next()
        img = decode_jpeg(sample['image'])
        pil_img = Image.fromarray(img, 'RGB')
        transformed_image = self.transform(pil_img)
        return transformed_image, sample['label'] 

    def __len__(self):
        return self.__len