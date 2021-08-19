#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:51:31 2021

On my existing conda env called segmenter, I built detectron2 from source. To do this, I first cloned our AR_Detectron repo:
    git clone https://github.com/kogans1107/AR_Detectron2.git
    
and then did the following pip install, which does a *LOT* of stuff I think. 

python -m pip install -e AR_Detectron2

THen I needed to install a python onnx package:
    
conda install -c conda-forge onnx=1.8.1

The newest version is 1.9, but this clashes with the Facebook export module. Facebook wants an onnx.optimizer, which has been removed from 1.9

After this, the following script demonstrates how to get a detectron2 class instance at the python prompt, and then how to export it to caffe2. I think. I am not yet sure precisely what I need to deliver to James and Jason. 

@author: bill
"""
import torch, torchvision
import detectron2
import onnx
# I don't know why they enable the logger in the colab version of this. I 
#   enabled it to start, then commented it out and it all seems fine. 
# from detectron2.utils.logger import setup_logger

# setup_logger()

import numpy as np
import cv2
import matplotlib.pyplot as plt
# import os, json, random   # I used none of these 

#  following requires me to be on colab I think, but regular cv2.imshow seems 
#   to work ok.  Well, it did for a while, argh
# from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def green_screen(img):
    black = np.sum(img,axis=2)==0
    glayer = img[:,:,1]
    glayer[black]=255
    img[:,:,1] = glayer
    return img

# may want the visualizer at some point
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

# Only need to run this once....uncomment this first tiume thru, then comment it
#   out forevermore.  
# !wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
# !mv input.jpg british_horse_dude.jpg

# im = cv2.imread("./british_horse_dude.jpg")
im = cv2.imread("./lab_objects.jpg")
# # OMG they broke imshow cv2.imshow('input image', im)
plt.figure(1)
plt.imshow(im)
imshape = im.shape

cfg = get_cfg()

#  Colab comments: 
# add project-specific config (e.g., TensorMask) here if you're not running a
#    model in detectron2's core library

# here we will need Karrington's model instead of this one. She knows how 
#   to make this replacement already. 
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/#mask_rcnn_R_50_FPN_3x.yaml")

#----------------------------
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("yolact_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 48

cfg.MODEL.WEIGHTS = 'model_final.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

#-----------------------------
predictor = DefaultPredictor(cfg)

# BOOM, drop the image through detectron2 here....
outputs = predictor(im)  # See? A neural net is just a function....

# Following shows some of how output is formatted...get the masks out and use them...
masksumpy = np.zeros(imshape[0:2])
nout = len(outputs['instances'])
for i in range(nout): # outputs['instances'] is not an iterable! HA!
    mask = outputs['instances'][i].get_fields()['pred_masks']  # sheesh!
    mm = mask.cpu().detach().numpy().reshape((imshape[0:2])).astype(np.uint8)
    masksumpy = np.logical_or(masksumpy, mm)
masksumpy = masksumpy.reshape((*imshape[0:2],1)).astype(np.uint8)

# OMG they broke imshow cv2.imshow('pytorch model', green_screen(im*masksumpy))

plt.figure(2)
plt.imshow(green_screen(im*masksumpy))

from detectron2 import export  # this is the tool for exporting models

model = predictor.model.eval()
if isinstance(model, torch.nn.modules.module.Module):
    print('model is of the correct type for export, hooray!')
else:
    print('model must be torch.nn.modules.module.Module, not', type(model))

neg = 255 - im  # just making a second image

#  inputs need to be in goofy, list of dicts format. With good old transposed 
#   pytorch shape....
image_list = [im, neg]
input_list = []
for image in image_list:
    input_list.append({'image': torch.from_numpy(image.transpose((2,0,1)))})

#
# Magic! Maybe it's even correct....need to check. Note that the outputs
#   will not be in the same format. The are "raw" and not "post-processed"
#   sayest the documentation. 
#   
ec2 = export.Caffe2Tracer(cfg, model, input_list)
caffe_model = ec2.export_caffe2()

caffe_model.save_protobuf('.') # saves several files to curr dir. Are they platform 
                            # dependent? They are protobuf, so should be ok


# The next line is possible because the object returned by export_caffe2() has 
#   a __call__ method defined. Which is super helpful for checking the model!
#
check = caffe_model([input_list[0]]) 
                                # I think the caffe2 model needs a list of dicts, 
                                # but the list can be only one element in length. 
                                #  BWAHAHAHAHA!
                                
                                
check_mask = check[0]['instances'].get_fields()['pred_masks'].cpu().detach().numpy()
check_mask = check_mask.astype(np.uint8)*255

masksum = np.zeros(imshape[0:2])
for i in range(len(check_mask)):
    masksum = np.logical_or(masksum,  check_mask[i,:,:].squeeze())
    # print(i,np.unique(check_mask[]))
masksum = masksum.reshape((*imshape[0:2],1)).astype(np.uint8)
# OMG they broke imshow cv2.imshow('caffe2 model',green_screen(im*masksum))
plt.figure(3)
plt.imshow(green_screen(im*masksum))


jff = ec2.export_onnx()
onnx.save_model(jff, './rename_this_sucker')