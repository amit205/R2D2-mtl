import os, pdb
import numpy as np
from PIL import Image
from .dataset import Dataset

class Kitti360Images_DB (Dataset):
    def __init__(self, root="data/kitti360"):
        Dataset.__init__(self)
        self.root = root
        self.imgs  = []
        self.semantics = []
        
        # try: 
        # read cached list
        img_list_path = os.path.join(self.root,"img_frames.txt")
        semantics_list_path = os.path.join(self.root,"semantics_frames.txt") 
        # cached_imgs = [e.strip() for e in open(img_list_path)]
        # cached_semantics = [e.strip() for e in open(semantics_list_path)]
        # assert cached_imgs, f"Cache '{img_list_path}' is empty!"
        # assert cached_semantics, f"Cache '{semantics_list_path}' is empty!"
        # self.imgs += cached_imgs
        # self.semantics += cached_semantics


        # except IOError:
                
        # create it
        imgs = []
        semantics = []
        folder = os.path.join(self.root, "train_images")
        folder_semantics = os.path.join(self.root, "masktrain2014")
        imgs = [f for f in os.listdir(folder) if verify_img(folder,f)]
        semantics = [f for f in os.listdir(folder_semantics) if verify_img(folder_semantics,f)]
        print(imgs[0])
        print(semantics[0])
        assert imgs, f"No images found in {folder}/"
        assert semantics, f"No images found in {folder_semantics}/"
        # open(img_list_path,'w').write('\n'.join(imgs))
        # open(semantics_list_path,'w').write('\n'.join(semantics))
        self.imgs = imgs
        self.semantics = semantics
        self.nimg = len(self.imgs)

    def get_key(self, i):
        key = self.imgs[i]
        return os.path.join('train_images',key)


def verify_img(folder, f):
    path = os.path.join(folder, f)
    if not (f.endswith('.png') or f.endswith('.jpg')): return False
    try: 
        from PIL import Image
        Image.open(path) # try to open it
        return True
    except: 
        return False