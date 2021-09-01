# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 05:58:55 2021

I wrote this so that I could experiment with a trained model from KTO and 
see if I could get detections, and then export it to ONNX with D415
native input image size (720, 1280) and get detections there in onnxruntime,
and then fingers crossed get detections in Acam2. 

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
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

import onnx, onnxruntime

from torch import nn, optim

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
        
        del boxes, labels, masks, areas, iscrowd, img_id, coco_annotation,\
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
    model = maskrcnn_resnet50_fpn(pretrained=False, \
                                  pretrained_backbone=False,\
                                  num_classes=49)
    
#    print('pick a weights file...')
#    sd = torch.load(uichoosefile()) 
    latest_weights_file = \
    'C:/Users/peria/Desktop/work/Brent Lab/git-repo/AR_Detectron2/save_models/CDC4-5-6-7-8-9CleanRedo89Ers_train.json-20210831_1722.pth'
    sd = torch.load(latest_weights_file)
    model.load_state_dict(sd)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    # create own Dataset
    my_dataset = CustomDataset(root=train_data_dir,
                              annotation=train_coco,
                              transforms=get_transform()
                              )

    batch_size = 1
    while True:
        try:
            data_loader = torch.utils.data.DataLoader(my_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)
            break
        except Exception as e:
            print('While making dataloader:\n',e)
            continue
 
          
    # DataLoader is iterable over Dataset
    data_loader_iterator = iter(data_loader)
    
    def get_a_batch():
        while True:
            try:
                imgs, annotations = data_loader_iterator.next()
                break
            except FileNotFoundError as e:  # I have some missing files
                print(e)
                continue
        image_list = list(img.to(device) for img in imgs)
        annotation_list = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        del imgs, annotations
        torch.cuda.empty_cache()
        return image_list, annotation_list

    image_list, annotations = get_a_batch()        

    native_size = (720,1280)
    go_native = torchvision.transforms.Resize(native_size)
    chk = model.eval()([go_native(image_list[0])])

    img_export = list(go_native(img).to('cpu') for img in image_list)[0]
    
    
    model_name = save_dir + 'MaskR-CNN_' + date_for_filename()
    onnx_name = model_name + '.onnx'

    torch.save(model.state_dict(), model_name + '_weights.pth')
        
    input_names = ['image']
    output_names = ['boxes','labels','scores','masks']
    dynamic_axes = {'boxes'  :  {0 : 'ndet'}, \
                    'labels' :  {0 : 'ndet'}, \
                    'scores' :  {0 : 'ndet'}, \
                    'masks'  :  {0 : 'ndet'}, \
                   }
    
    torch.onnx.export(model.to('cpu').eval(), \
                      [img_export], \
                      onnx_name, \
                      export_params=True, \
                      do_constant_folding=True, \
                      verbose=True, \
                      opset_version=11, \
                      input_names=input_names, \
                      output_names=output_names, \
                      dynamic_axes=dynamic_axes)

#'image'  :  {1 : 'width', 2 : 'height'}
    ort_session = onnxruntime.InferenceSession(onnx_name)
    input_name = ort_session.get_inputs()[0].name

    real_image = go_native(image_list[0])
    ort_inputs = {input_name: to_numpy(real_image)}
    ort_outs = ort_session.run(None, ort_inputs)

    isize = real_image.size()
    itype = real_image.dtype
    rand_image = torch.randn(isize, dtype=itype)
    rand_inputs = {input_name: to_numpy(rand_image)}
    rand_outs = ort_session.run(None, rand_inputs)

    plt.figure(1)
    plt.imshow(np.transpose(real_image.cpu(), axes=[1,2,0]))

    masks = ort_outs[3]
    for i,mask in enumerate(masks):
        plt.figure(2+i)
        plt.imshow(mask[0,:,:])
    
    
# 'C:\\Users\\peria\\Desktop\\work\\Brent Lab\\git-repo\\AR_Detectron2\\save_models\\MaskR-CNN_20210830_2026.onnx'
    