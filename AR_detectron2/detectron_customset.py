#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:19:29 2021

@author: karrington
"""
# install detectron2: (Colab has CUDA 10.1 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
assert torch.__version__.startswith("1.8")   # need to manually install torch 1.8 if Colab changes its default version
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output


import json
import tkinter as tk
from tkinter import filedialog
import os
import time
import numpy as np
from pathlib import Path


def uichoosefile(title = None, initialdir = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename(title=title, initialdir = initialdir)
    
    
# def my_dataset_function():
#   ...
#   return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
# later, to access the data:
data: List[Dict] = DatasetCatalog.get("my_dataset")


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset'_train", {}, "json_annotation_train.json", "path/to/image/dir")
register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")


def my_dataset_function():
    data_path = 'coco_yolact_data/'
    train_set = uichoosefile(title='Choose train JSON file...')
    val_set = uichoosefile(title='Choose val JSON file...')
    
    tr = train_set.split(sep='/')
    val = val_set.split(sep='/')
    
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances(Path(tr[-1]).stem, {}, data_path + tr[-2] + '/' +  tr[-1], data_path + 'images/' )
    register_coco_instances(Path(val[-1]).stem, {}, data_path +  val[-2] + '/' +  val[-1], data_path + 'images/' )






def get_dicts(json_name):
    json_file = os.path.join('coco_yolact_data/annotations',json_name)
    with open(json_file) as f:
        json_set = json.load(f)

    dataset_dicts = []
    for v in json_set['images']:
        record = {}
        
        filename = os.path.join('coco_yolact_data/images', v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


dataset_dicts = get_dicts('CDC9_Test_Val.json')
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=yolact_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])



# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

    
def evaluations():
    
    cfg = get_cfg()
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    
    
    from detectron2.utils.visualizer import ColorMode
    
    
    file1 = uichoosefile()
    
    with open(file1,'r') as fp:
           dataset_dicts = json.load(fp)
            
    os.chdir('/media/karrington/FantomHD/AR_yolact/data/coco/images')
    
    save_file = '/media/karrington/FantomHD/detectron2/evaluations/'
    for d in random.sample(dataset_dicts['images'], 3):
        
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=yolact_metadata, 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('new', out.get_image()[:, :, ::-1])
        cv2.waitKey(5000)
        #cv2.imwrite(save_file + d['file_name'],out)
        
    os.chdir('/media/karrington/FantomHD/detectron2/')
    
    
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("yolact_train",)
cfg.DATASETS.TEST = ("yolact_val")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 38  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode
error_list = ()
def get_data_dicts(img_dir):
    # os.chdir('/media/karrington/FantomHD/AR_yolact/data/coco/images')
    # img_dir = '/media/karrington/FantomHD/AR_yolact/data/coco/images'
    json_file = uichoosefile()
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    ann_len = len(imgs_anns['annotations'])
    
    print('len of ann:', ann_len)
    for idx, a in enumerate(imgs_anns['annotations']):
        record = {}
        if idx % 100 == 0:
            print(ann_len - idx, 'left')
        try:
            img_id = a['image_id']
            images = imgs_anns['images']
            filename = os.path.join(img_dir, images[img_id]["file_name"])
            # height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = img_id
            record["height"] = images[img_id]['height']
            record["width"] = images[img_id]['width']
           
          
      
            objs = []
    
            obj = {
                "bbox": a['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": a['segmentation'],
                "category_id": a['category_id'],
                }
            objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        except IndexError:
            error_list.append(img_id)
            #print('error',img_id)
 
    #os.chdir('/media/karrington/FantomHD/detectron2/')
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")









