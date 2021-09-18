# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:20:24 2021

I am trying to see whether the rotation of the mask relative to the image 
is causing a problem in training. 

@author: Bill
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:46:28 2021

Here I combined Karrington's custom data loader with the ONNX exporting of 
    Mask R-CNN. It successfully loaded into Jasper's test loader. 
    
When the model is in eval mode, it returns a list of dicts with keys:
    dict_keys(['boxes', 'labels', 'scores', 'masks'])
    
When the model is in train mode, we can't call it yet. It needs masks and 
    does not have them. 

@author: karrington, bill
"""


import os
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image
from pycocotools.coco import COCO

import tkinter as tk
from tkinter.filedialog import FileDialog

def uichoosefile(title = None, initialdir = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename(title=title, initialdir = initialdir)
    return filename


#  Uncomment for Karrington version
#train_data_dir = '/media/karrington/FantomHD/coco/images'
#train_coco = '/media/karrington/FantomHD/coco/annotations/CleanCombo2.0_coco_train.json'
#save_dir = ???

# Uncomment for Bill version
os.chdir('../AR_yolact/data/coco')
train_data_dir = 'images'
train_coco = 'annotations/TDS4-5-6_4.0_train.json'
save_dir = 'C:\\Users\\peria\\Desktop\\work\\Brent Lab\\git-repo\\AR_Detectron2\\save_models\\'

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
#        print('opening file:', os.path.join(self.root, path))
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for cai in coco_annotation:
            xmin = cai['bbox'][0]
            ymin = cai['bbox'][1]
            width = cai['bbox'][2]
            height = cai['bbox'][3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = []        
        for cai in coco_annotation:
            labels.append(torch.tensor(cai['category_id']))
        labels = torch.tensor(labels, dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        areas = []
        for cai in coco_annotation:
            areas.append(cai['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)


        masks = []
        for ann in coco_annotation:
            this_mask = coco.annToMask(ann)
#            this_mask = [coco.annToMask(obj).reshape(-1) for obj in target]
            masks.append(this_mask)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["masks"] = masks
        
        del boxes, labels, masks, areas, iscrowd, img_id, coco_annotation,
        this_mask
        torch.cuda.empty_cache()

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
    

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def date_for_filename():
    tgt = time.localtime()
    year = str(tgt.tm_year)
    mon = "{:02}".format(tgt.tm_mon)
    day = "{:02}".format(tgt.tm_mday)
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    datestr = year + mon + day + '_' + hour + minute
    return datestr

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



if __name__=="__main__":
    


    # create own Dataset
    my_dataset = CustomDataset(root=train_data_dir,
                              annotation=train_coco,
                              transforms=get_transform()
                              )

    # Batch size
    train_batch_size = 1
    # own DataLoader. num_workers=0 is essential on Windows to avoid 
    #   the ForkingPickler error. 
    while True:
        try:
            data_loader = torch.utils.data.DataLoader(my_dataset,
                                                      batch_size=train_batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)
            break
        except Exception as e:
            print('While making dataloader:\n',e)
            continue
 
       
    
    data_loader_iterator = iter(data_loader)
    
# Karrington, if you put this next block in a loop, you could let it run 
#    until it finds a messed up mask, i.e. put a break in the else clause.
# For me it does not take long to find one; probably same for you. 
    
    def hw_match(img1, img2):
        return img1.shape[0:2] == img2.shape[0:2]
        
    
    fig,ax = plt.subplots(1,2)

    for image_list, annotations in data_loader:
        try:
            which_image = annotations[0]['image_id'].item()
            if which_image > 100:
                break
            
            imgshow=np.transpose(image_list[0].cpu(), axes=(1,2,0))
        
            maskshow = annotations[0]['masks'][0,:,:].cpu() # I luv computers. So clear!
            if hw_match(imgshow,maskshow):  # shapes match, whew...
                pass
            else:  # ACK! Shapes do not match!!!
                fixmask = np.transpose(np.fliplr(maskshow)) # this will make them 
                maskshow = fixmask                         #   in cases I've seen
    
            ax[0].imshow(imgshow)
            ax[0].set_title(str(which_image))
            ax[1].imshow(imgshow*maskshow.reshape((*maskshow.shape,1)))
            
            
            plt.pause(0.1)
        except Exception as e:
            print('Image',which_image,'has a problem...')
            print(e)
     
    
    