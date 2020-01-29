# 3D-Unet: patched based Keras implementation for medical images segmentation

3D-Unet pipeline is a computational toolbox (python-Keras) for segmentation using neural networks. 

![3D U-net](images/unet.png)

The training and the inference are patch based: the script randomly extract corresponding patches of the images and labels and feed them to the network during training.
The inference script extract, segment the sigle patches and automatically recontruct them in the original size.

### Example images

Sample MR images from the sagittal and coronal views for carotid artery segmentation (the segmentation result is highlighted in green)

![MR3](images/3.JPG)![MR4](images/4.JPG)
*******************************************************************************

### Requirements
- Python3
- pillow
- scikit-learn
- simpleITK
- keras
- scikit-image
- pandas
- pydicom
- nibabel
- tqdm
- git+https://www.github.com/farizrahman4u/keras-contrib.git

### Python scripts and their function

- generator.py / NiftiDataset.py : They augment the data, extract the patches and feed them to the GAN (reads .nii files). NiftiDataset.py
  skeleton taken from https://github.com/jackyko1991/vnet-tensorflow

- check_loader_patches: Shows example of patches fed to the network during the training  

- unet3d.py: the architecture of the U-net. Taken from https://github.com/ellisdg/3DUnetCNN

- metrics.py : list of metrics and loss functions for the training

- main.py: Runs the training and the prediction on the training and validation dataset.

- predict.py: It launches the inference on training and validation data in the main.py

- segment_single_image.py: It launches the inference on a single input image chosen by the user.

## Usage
As example, run the following command to start the training:
```console
python3 main.py --Create_training_val_test_dataset=True --Do_you_wanna_train=True --save_dir ./pathtoimageslabelsfolders --images_folder= ./pathtotheimages --labels_folder= ./pathtothelabels --patch_size=(64,64,64)
```
There are several parameters you need to set; you can modify the default ones in the script or write them in order manually:

- Use_GPU, help='Use the GPU'
- Select_GPU, help='Select the GPU'
- Create_training_val_test_dataset, help='Divide the data for the training'
- Do_you_wanna_train, help='Training will start'
- Do_you_wanna_load_weights, help='Pre-load weights'
- Do_you_wanna_check_accuracy, help='Model will be tested'
- save_dir, help='path to the images and labels folders'
- images_folder, help='path to the .nii images'
- labels_folder, help='path to the .nii labels'
- val_split, help='Split value for the validation data (0 to 1 float number)'
- test_split, help='Split value for the test data (0 to 1 float number)'
- history_dir, help='path where to save sample images during training'
- weights, help='path to save the weights of the model'
- unet_weights, help='unet weights to load/save'

- resample, help='Decide or not to resample the images to a new resolution'
- new_resolution, help='New image resolution'
- input_channels, help="Input channels"
- output_channels,help="Output channels"
- patch_size, help="Input dimension for the network"
- batch_size, help="Batch size to feed the network (currently supports 1)"
- drop_ratio, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1"
- min_pixel, type=float, nargs=1, default=0.001, help="Percentage of minimum non-zero pixels in the cropped label"
- labels, help="Values of the labels"
- n_labels, help="The label numbers on the input image"
- initial_learning_rate, help="learning rate"
- nb_epoch, help="number of epochs"
- patience, help="learning rate will be reduced after this many epochs if the validation loss is not improving"
- early_stop, help="training will be stopped after this many epochs without the validation loss improving"
- learning_rate_drop, help="factor by which the learning rate will be reduced"
- n_images_per_epoch, help="Number of training images per epoch"

- stride_inplane, help="Stride size in 2D plane"
- stride_layer, help="Stride size in z direction"

For further details you can open each script and read the description and list of commands.

## Features
- 3D data processing ready
- Augmented patching technique, requires less image input for training
- Multichannel input and one channel output (multichannel to be developed)
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Dice score similarity measurement as golden standard in medical image segmentation benchmarking
- Includes Tensorboard to track the training process

## Citations
Use the following Bibtex if you need to cite this repository:
```bibtex
@misc{davidiommi1991_unet_Keras,
  author = {David Iommi},
  title = {3D-Unet: patched based Keras implementation for medical images segmentation},
  howpublished = {\url{https://github.com/davidiommi/3D-U-net-Keras}},
  year = {2020},
  publisher={Github},
  journal={GitHub repository},
}

@misc{jackyko1991_vnet_tensorflow,
  author = {Jacky KL Ko},
  title = {Implementation of vnet in tensorflow for medical image segmentation},
  howpublished = {\url{https://github.com/jackyko1991/vnet-tensorflow}},
  year = {2018},
  publisher={Github},
  journal={GitHub repository},
}