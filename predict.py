#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from utils.NiftiDataset import *
from utils.generator import *
from utils.metrics import *
from models_networks.unet3d import *
from training_unet3d import *
from tqdm import tqdm
import datetime
from segment_single_image import *


def inference_all(model, image_path, label_path, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    case = image_path          # dgx linux
    case = case.split('/')
    case = case[3]
    case = case.split('.')
    case = case[0]

    # case = image_path        # laptop
    # case = case.split('/')
    # case = case[2]
    # case = case.split('.')
    # case = case[0]
    # case = case.split('\\')
    # case = case[1]

    if not os.path.isdir('./Data_folder/results'):
        os.mkdir('./Data_folder/results')

    label_directory = os.path.join('./Data_folder/results', case)

    if not os.path.isdir(label_directory):  # create folder
        os.mkdir(label_directory)

    result, np_dice, ravd, hauss_dist, mean_surf_dist = inference(write_image=False, model=model, image_path=image_path, label_path=label_path,
                                                                  result_path='./prova.nii', resample=resample, resolution=resolution, patch_size_x=patch_size_x,
                      patch_size_y=patch_size_y, patch_size_z=patch_size_z, stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=batch_size)

    # save segmented label
    writer = sitk.ImageFileWriter()
    label_directory = os.path.join(label_directory, 'label_prediction.nii.gz')
    writer.SetFileName(label_directory)
    writer.Execute(result)
    # print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_path))
    print('************* Next image coming... *************')

    return result, np_dice, ravd, hauss_dist, mean_surf_dist


def check_accuracy_model(model, images_list, labels_list, resample, new_resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    model = model

    f = open(images_list, 'r')
    images = f.readlines()
    f.close()

    f = open(labels_list, 'r')
    labels = f.readlines()
    f.close()

    np_dice = []
    ravd = []
    hauss_dist = []
    mean_surf_dist = []

    print("0/%i (0%%)" % len(labels))
    for i in range(len(labels)):

        result, Np_dice, Ravd, Hauss_dist, Mean_surf_dist = inference_all(model=model, image_path=images[i].rstrip(), label_path=labels[i].rstrip(), resample=resample, resolution=new_resolution,
                                                                  patch_size_x=patch_size_x, patch_size_y=patch_size_y, patch_size_z=patch_size_z,  stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=batch_size)

        np_dice.append(Np_dice)
        ravd.append(Ravd)
        hauss_dist.append(Hauss_dist)
        mean_surf_dist.append(Mean_surf_dist)

    np_dice = np.array(np_dice)
    ravd = np.array(ravd)
    hauss_dist = np.array(hauss_dist)
    mean_surf_dist = np.array(mean_surf_dist)

    print('Mean volumetric DSC:', np_dice.mean())
    print('Std volumetric DSC:', np_dice.std())
    print('Median volumetric DSC:', np.median(np_dice))
    print('Mean Rel. Abs. Vol. Diff:', ravd.mean())
    print('Mean Hauss. Dist:', np.mean(hauss_dist))
    print('Mean MSD:', np.mean(mean_surf_dist))



