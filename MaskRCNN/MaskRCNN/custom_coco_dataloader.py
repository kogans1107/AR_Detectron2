#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:46:28 2021

@author: karrington
"""


import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import cv2 

import pandas as pd

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

YOLAB_LABEL_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48}

YOLAB_CLASSES = ('8-Cap Strip Bag', '96 Well Cooling Block', '96 Well PCR Plate', 'Bunsen Burner', '15ml Capped Tube', 'Gloves', 'Vortexer', 'Vortex', 'Hand', 'Ice Bucket', 'GeneDrive', 'PCR Machine', 'Laptop', 'M', 'Micropipette', 'Micropipette Tip', 'N1', 'N2', 'NA', 'Nuclease Free PCR Water', 'Nucleic Acid Decontaminant', 'Spark Striker', 'Opaque Rack Cover', 'VWR Marker', 'PC', 'Microcentrifuge', 'Pipette Tip Box', 'R', 'RP', 'Rack for 1.5-2ml Tubes', 'S', 'Sterile 1.5mL Tubes', 'Sterile 1.5mL Tubes', 'Thumb', 'Trash', 'Tube Rack', 'Electronic Pipet-Aid')

# def get_label_map():
#     if cfg.dataset.label_map is None:
#         return {x+1: x+1 for x in range(len(classname))}
#     else:
#         return cfg.dataset.label_map 

# class COCOAnnotationTransform(object):
#     """Transforms a COCO annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes
#     """
#     def __init__(self):
#         self.label_map = YOLAB_LABEL_MAP

#     def __call__(self, target, width, height):
#         """
#         Args:
#             target (dict): COCO target json annotation as a python dict
#             height (int): height
#             width (int): width
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class idx]
#         """
#         scale = np.array([width, height, width, height])
#         res = []
#         for obj in target:
#             if 'bbox' in obj:
#                 bbox = obj['bbox']
#                 label_idx = obj['category_id']
#                 if label_idx >= 0:
#                     label_idx = self.label_map[label_idx] - 1
#                 final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
#                 final_box.append(label_idx)
#                 res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
#                 #print(type(res))
#             else:
#                 print("No bbox found for object ", obj)

#         return torch.FloatTensor(res)



class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None,has_gt=True):
        self.root = root
        self.transforms = transforms
        #self.target_transforms = COCOAnnotationTransform()
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.has_gt = has_gt

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        
        if self.has_gt:
            #target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
        else:
            target = []

        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        # boxes = []
        # for i in range(num_objs):
        #     xmin = coco_annotation[i]['bbox'][0]
        #     ymin = coco_annotation[i]['bbox'][1]
        #     xmax = xmin + coco_annotation[i]['bbox'][2]
        #     ymax = ymin + coco_annotation[i]['bbox'][3]
        #     # width = coco_annotation[i]['bbox'][2]
        #     # height = coco_annotation[i]['bbox'][3]
        #     boxes.append([xmin, ymin, xmax, ymax])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
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
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
#         masks = []
#         for ann in coco_annotation:
#             this_mask = coco.annToMask(ann)
# #            this_mask = [coco.annToMask(obj).reshape(-1) for obj in target]
#             masks.append(this_mask)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        imgshow = np.transpose(img, axes=(1,2,0))

        masks = []
        #print(len(coco_annotation))
        for ann in coco_annotation:
            this_mask = coco.annToMask(ann)
            if imgshow.shape[0:2]==this_mask.shape:
                masks.append(this_mask)
            else:
                #print('imshow: ',imgshow.shape)
                #print('this_mask: ',this_mask.shape)
                fixedmask = np.transpose(np.fliplr(this_mask))
                #print('fixedmask:',fixedmask.shape)
                masks.append(fixedmask)
#            this_mask = [coco.annToMask(obj).reshape(-1) for obj in target]
            #masks.append(this_mask)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # # Iscrowd
        # iscrowd = []
        # for i in range(num_objs):
        #     iscrowd.append(coco_annotation[i]['iscrowd'])
        # iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
        
         # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # category_id = []     
        # for i in range(num_objs):
        #     category_id.append(coco_annotation[i]['category_id'])
        # category_id = torch.as_tensor(category_id, dtype=torch.int64)
        
        # mask = [self.coco.annToMask(obj) for obj in target]
        # # mask = np.vstack(mask)
        # # #mask = torch.as_tensor(mask, dtype=torch.int64)
        # # mask = mask.reshape(-1, int(height), int(width))
        # mask = torch.as_tensor(mask, dtype=torch.float32)
        
        # for i in range(num_objs):
        #     width = coco_annotation[i]['bbox'][2]
        #     height = coco_annotation[i]['bbox'][3]
        #     target[i] = self.target_transforms(target[i], width, height)
        
        # mask = []
        # for i in range(num_objs):
        #     for j in range(len(coco_annotation[i]['segmentation'][0]))
        #         mask.append(self.coco.annToMask(coco_annotation[i]['segmentation'][0][j]))
        # segmentation = []
        # for i in range(num_objs):
        #       segmentation.append(coco_annotation[i]['segmentation'])
        # segmentation = torch.as_tensor(segmentation, dtype=torch.float32)
        
        # ids = []
        # for i in range(num_objs):
        #     ids.append(coco_annotation[i]['id'])
        # ids = torch.as_tensor(ids, dtype=torch.float32)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation['masks'] = masks
        # my_annotation['segmentation'] = segmentation
        #my_annotation['labels'] = category_id
        # my_annotation['id'] = ids
        
        #my_annotation["targets"] = target

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
    

train_data_dir = '/media/karrington/FantomHD/coco/images'
train_coco = '/home/karrington/Downloads/CDC4-5-6-7-8-9(CleanRedo89Ers)_train.json'
test_coco = '/home/karrington/Downloads/CDC4-5-6-7-8-9(CleanRedo89Ers)_val.json'
dataset_name = train_coco.split(sep='/')[-1].split(sep='.')[0].split(sep='_train')[0] + '_'
save_path = '/home/karrington/git.workspace/vision/weights/' + '_'
def get_transform(train):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    if train:
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
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

if __name__=="__main__":
    model = maskrcnn_resnet50_fpn(pretrained=False, \
                              pretrained_backbone=False,\
                              num_classes=49)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #model.load_state_dict(torch.load('/home/karrington/FantomHD/CDC4-5-6-7-8-9(CleanRedo89Ers)_train-json-20210831_1301.pth'))
    
    model = model.to(device)
    
    # create test and train dataset
    dataset = myOwnDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform(train=True)
                          )
    dataset_test = myOwnDataset(root=train_data_dir,
                          annotation=test_coco,
                          transforms=get_transform(train=False)
                          )
   #split datasets
   
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-40])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-40:])
    
    # Batch size
    train_batch_size = 2

# own DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)
        
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=train_batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=collate_fn)
    
    len_dataloader = len(data_loader)
    
    #data_loader_iterator = iter(data_loader)
    
    loader_iter = iter(data_loader)
    
    def get_a_batch():
        loader_iter = iter(data_loader)
        while True:
          try:
              imgs, annotations = next(loader_iter)
              break
          except StopIteration:
              loader_iter = iter(data_loader)
              imgs, annotations = next(loader_iter)
              
          except FileNotFoundError as e:  # I have some missing files
              print(e)
              continue
            
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        return imgs,annotations

        
    
    # optimizer_type = optim.Adam
    # optimizer = optimizer_type(model.parameters(), lr=1e-5)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
    print('...calling model in training mode...')
    num_epochs = 10
    for epoch in range(num_epochs):
   # train for one epoch, printing every 10 iterations
       train_one_epoch(model, optimizer, data_loader, device, epoch,
                   print_freq=10)# update the learning rate
       lr_scheduler.step()
   # evaluate on the test dataset
       evaluate(model, data_loader_test, device=device)
       torch.save(save_path + dataset_name + epoch + date_for_filename())
    
    
    # for i in range(10000):
    #     # imgs, annotations = data_loader_iterator.next()
        
    #     # imgs = list(img.to(device) for img in imgs)
    #     # annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #     imgs,annotations = get_a_batch()
        
    #     tout = model.train()(imgs, annotations)
    #     optimizer_type = optim.Adam
    #     optimizer = optimizer_type(model.parameters(), lr=1e-5)
    #     optimizer.zero_grad() 
    
    #     loss = torch.tensor(0.0).cuda()
    #     for k,v in tout.items():
    #         loss += v
            
    #     loss.backward()
    #     optimizer.step()
    #     #print(loss)
    #     epoch = i//len_dataloader
    #     if i%10 == 0:
    #         print(f'Epoch:{epoch},Iteration: {i}/{len_dataloader}, Loss: {loss}')
    #     if i%1000 == 0:
    #         print('saving {i} iteration to weights')
    #         torch.save(model.state_dict(),train_coco.split(sep='/')[-1] +'-' + date_for_filename() + '.pth')
        
    # img_exp = list(img.to('cpu') for img in imgs)
    
    # torch.onnx.export(model.to('cpu').eval(), img_exp,\
    #                   'MaskR-CNN_' + date_for_filename() + '.onnx',\
    #                   verbose=True, opset_version=11)
    # #imgs = list(img.to(device) for img in imgs)

#DataLoader is iterable over Dataset
# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #print(((annotations[0]['boxes'])))



# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = maskrcnn_resnet50_fpn(pretrained=False, \
#                               pretrained_backbone=False,\
#                               num_classes=49)
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = maskrcnn_resnet50_fpn(in_features, num_classes)
    
#     return model

# num_classes = 49
# num_epoch = 10

# model = maskrcnn_resnet50_fpn(pretrained=False, \
#                               pretrained_backbone=False,\
#                               num_classes=49)

#model = get_model_instance_segmentation(num_classes)

# model.to(device)

# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# len_dataloader = len(data_loader)

# for epoch in range(num_epoch):
#     model.train()
#     i = 0    
#     for imgs, annotations in data_loader:
#         i += 1
#         imgs = list(img.to(device) for img in imgs)
#         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#         loss_dict = model(imgs, annotations)
#         losses = sum(loss for loss in loss_dict.values())
        
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
        
#         print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
    
