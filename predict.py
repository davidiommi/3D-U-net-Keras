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


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches


# segment single image
def segment_image_evaluate(model, image_path, label_path, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    # create transformations to image and labels
    transforms = [
        NiftiDataset.Resample(resolution, resample),
        NiftiDataset.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    case = image_path
    case = case.split('/')
    case = case[3]
    case = case.split('.')
    case = case[0]

    if not os.path.isdir('./Data_folder/segmentation_results'):
        os.mkdir('./Data_folder/segmentation_results')

    label_directory = os.path.join('./Data_folder/segmentation_results', case)

    if not os.path.isdir(label_directory):  # create folder
        os.mkdir(label_directory)

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    image = Normalization(image)  # set intensity 0-255

    # read label file
    reader = sitk.ImageFileReader()
    reader.SetFileName(label_path)
    label = reader.Execute()

    # preprocess the image and label before inference
    image_tfm = image

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image_tfm.GetSize(), sitk.sitkUInt8)
    label_tfm.SetOrigin(image_tfm.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image_tfm.GetSpacing())

    original = {'image': image_tfm, 'label': label}
    sample = {'image': image_tfm, 'label': label_tfm}

    for transform in transforms:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']
    label_true = original['label']

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    image_np = np.asarray(image_np, np.uint8)

    label_np = sitk.GetArrayFromImage(label_tfm)
    label_np = np.asarray(label_np, np.float32)

    label_true_np = sitk.GetArrayFromImage(label_true)
    label_true_np = np.asarray(label_true_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)

    # acutal segmentation
    for i in tqdm(range(len(batches))):
        batch = batches[i]

        pred = model.predict(batch, verbose=2, batch_size=1)  # predict segmentation
        pred = np.squeeze(pred, axis=4)

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[0, :, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0


    print("{}: Evaluation complete".format(datetime.datetime.now()))
    # eliminate overlapping region using the weighted value
    label_np = np.rint(np.float32(label_np) / np.float32(weight_np) + 0.01)

    # convert back to sitk space
    label_np = np.transpose(label_np, (2, 1, 0))

    # convert label numpy back to sitk image
    label_tfm = sitk.GetImageFromArray(label_np)
    label_tfm.SetOrigin(image_tfm.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image_tfm.GetSpacing())

    # save segmented label
    writer = sitk.ImageFileWriter()

    if resample is True:

        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        label = resample_sitk_image(label_tfm, spacing=image.GetSpacing(), interpolator='nearest')
        label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkNearestNeighbor)
        label_np = sitk.GetArrayFromImage(label)[::-1]
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())

    else:
        label = label_tfm

    label_directory = os.path.join(label_directory, 'label_prediction.nii.gz')
    writer.SetFileName(label_directory)
    writer.Execute(label)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_path))

    print('Dice prediction:', numpy_dice(label_true_np, label_np))
    print('Hauss. Dist:', (surface_dist(label_true_np, label_np)).max())

    np_dice = numpy_dice(label_true_np, label_np, axis=None)
    ravd = rel_abs_vol_diff(label_true_np, label_np)
    hauss_dist = (surface_dist(label_true_np, label_np)).max()
    mean_surf_dist = (surface_dist(label_true_np, label_np)).mean()

    return np_dice, ravd, hauss_dist, mean_surf_dist


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

        Np_dice, Ravd, Hauss_dist, Mean_surf_dist = segment_image_evaluate(model= model, image_path=images[i].rstrip(), label_path=labels[i].rstrip(),
                                                                           resample= resample, resolution=new_resolution, patch_size_x=patch_size_x,
                                        patch_size_y=patch_size_y, patch_size_z=patch_size_z, stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=batch_size)

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



