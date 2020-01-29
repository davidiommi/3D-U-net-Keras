import os
import time
import pandas as pd
import argparse
from utils.NiftiDataset import *
from utils.metrics import *
from utils.generator import *
from models_networks.unet3d import *
from training_unet3d import *
from predict import *

start = time()

# -----------------
# CONFIGURATIONS
# -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--Use_GPU", action='store_true', default=True, help='Use the GPU')
parser.add_argument("--Select_GPU", type=int, default=2, help='Select the GPU')
parser.add_argument("--Create_training_val_test_dataset", action='store_true', default=False, help='Divide the data for the training')
parser.add_argument("--Do_you_wanna_train", action='store_true', default=False, help='Training will start')
parser.add_argument("--Do_you_wanna_load_weights", action='store_true', default=False, help='Load weights')
parser.add_argument("--Do_you_wanna_check_accuracy", action='store_true', default=True, help='Model will be tested after the training')
parser.add_argument("--save_dir", type=str, default='./Data_folder/', help='path to the images folders')
parser.add_argument("--images_folder", type=str, default='./Data_folder/volumes', help='path to the .nii images')
parser.add_argument("--labels_folder", type=str, default='./Data_folder/labels', help='path to the .nii labels')
parser.add_argument("--val_split", type=float, default=0.1, help='Split value for the validation data (0 to 1 float number)')
parser.add_argument("--test_split", type=float, default=0.1, help='Split value for the test data (0 to 1 float number)')
parser.add_argument("--history_dir", type=str, default='./History', help='path where to save sample images during training')
parser.add_argument("--weights", type=str, default='./History/weights', help='path to save the weights of the model')
parser.add_argument("--unet_weights", type=str, default='./History/weights/unet3d.h5', help='unet weights to load/save')
# Training parameters
parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(0.9375, 0.9375, 3.0), help='New resolution')
parser.add_argument("--input_channels", type=float, nargs=1, default=1, help="Input channels")
parser.add_argument("--output_channels", type=float, nargs=1, default=1, help="Output channels")
parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128], help="Input dimension for the network")
parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
parser.add_argument("--drop_ratio", type=float, nargs=1, default=0.5, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1")
parser.add_argument("--min_pixel", type=float, nargs=1, default=0.001, help="Percentage of minimum non-zero pixels in the cropped label")
parser.add_argument("--labels", type=int, default=[1], help="Values of the labels")
parser.add_argument("--n_labels", type=int, default=1, help="The label numbers on the input image")
parser.add_argument("--initial_learning_rate", type=float, nargs=1, default=0.0002, help="learning rate")
parser.add_argument("--nb_epoch", type=int, nargs=1, default=200, help="number of epochs")
parser.add_argument("--patience", type=int, nargs=1, default=10, help="learning rate will be reduced after this many epochs if the validation loss is not improving")
parser.add_argument("--early_stop", type=int, nargs=1, default=41, help="training will be stopped after this many epochs without the validation loss improving")
parser.add_argument("--learning_rate_drop", type=float, nargs=1, default=0.5, help="factor by which the learning rate will be reduced")
parser.add_argument("--n_images_per_epoch", type=int, nargs=1, default=100, help="Number of training images per epoch")
# Inference parameters
parser.add_argument("--stride_inplane", type=int, nargs=1, default=64, help="Stride size in 2D plane")
parser.add_argument("--stride_layer", type=int, nargs=1, default=64, help="Stride size in z direction")
args = parser.parse_args()

if args.Use_GPU is True:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Select_GPU)

if not os.path.exists(args.history_dir):
    os.makedirs(args.history_dir)
if not os.path.exists(args.weights):
    os.makedirs(args.weights)

min_pixel = int(args.min_pixel * ((args.patch_size[0] * args.patch_size[1] * args.patch_size[2]) / 100))

# ---------------------------------------------------------
# 1) Create training, test, validation data-set text lists
# ---------------------------------------------------------

images = lstFiles(args.images_folder)
labels = lstFiles(args.labels_folder)

if args.Create_training_val_test_dataset is True is True:

    images, test_images, labels, test_labels = split_train_set(args.test_split, images, labels)
    images, val_images, labels, val_labels = split_train_set(args.val_split, images, labels)

    write_list(args.save_dir + '/' + 'train.txt', images)
    write_list(args.save_dir + '/' + 'train_labels.txt', labels)
    write_list(args.save_dir + '/' + 'val.txt', val_images)
    write_list(args.save_dir + '/' + 'val_labels.txt', val_labels)
    write_list(args.save_dir + '/' + 'test.txt', test_images)
    write_list(args.save_dir + '/' + 'test_labels.txt', test_labels)


# ------------------------
# 2) Training model
# ------------------------

if args.Do_you_wanna_train is True:

    f = open(args.save_dir + '/' + 'train.txt', 'r')
    n_samples_train = len(f.readlines())
    f.close()

    f = open(args.save_dir + '/' + 'val.txt', 'r')
    n_samples_val = len(f.readlines())
    f.close()

    f = open(args.save_dir + '/' + 'test.txt', 'r')
    n_samples_test = len(f.readlines())
    f.close()

    print('Number of training samples:', n_samples_train, '  Number of validation samples:', n_samples_val, '  Number of testing samples:', n_samples_test)

    trainTransforms = [
        NiftiDataset.Resample(args.new_resolution, args.resample),
        NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
        NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio, min_pixel),
        NiftiDataset.Augmentation(),
    ]

    valTransforms = [
        NiftiDataset.Resample(args.new_resolution, args.resample),
        NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
        NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
                                min_pixel),
    ]

    train_gen = data_generator(images_list=args.save_dir + '/' + 'train.txt', labels_list=args.save_dir + '/' + 'train_labels.txt',
                         batch_size=args.batch_size, Transforms=trainTransforms)
    val_gen = data_generator(images_list=args.save_dir + '/' + 'val.txt', labels_list=args.save_dir + '/' + 'val_labels.txt',
                       batch_size=args.batch_size, Transforms=valTransforms)

    # instantiate new model

    size_input = (args.batch_size, args.patch_size[0], args.patch_size[1], args.patch_size[2], args.input_channels)

    model = unet_model_3d(input_shape=size_input, n_labels=args.n_labels, initial_learning_rate=args.initial_learning_rate)

    if args.Do_you_wanna_load_weights is True:
        model.load_weights(args.unet_weights)

    model.fit_generator(generator=train_gen, steps_per_epoch=args.n_images_per_epoch, epochs=args.nb_epoch,
                        validation_data=val_gen, validation_steps=args.n_images_per_epoch/10,
                        callbacks=get_callbacks(args.unet_weights,
                                                initial_learning_rate=args.initial_learning_rate,
                                                learning_rate_drop=args.learning_rate_drop,
                                                learning_rate_epochs=args.patience,
                                                early_stopping_patience=args.early_stop))

    model.save_weights(args.unet_weights)

# ----------------------------
# 3) Check accuracy model
# ----------------------------

if args.Do_you_wanna_check_accuracy is True:

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')

    size_input = (args.batch_size, args.patch_size[0], args.patch_size[1], args.patch_size[2], args.input_channels)
    model = unet_model_3d(input_shape=size_input, n_labels=args.n_labels)
    model.load_weights(args.unet_weights)

    check_accuracy_model(model, images_list=args.save_dir + '/' + 'val.txt', labels_list=args.save_dir + '/' + 'val_labels.txt', resample=args.resample,
                         new_resolution=args.new_resolution, patch_size_x=args.patch_size[0],patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],
                         stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1)

    check_accuracy_model(model, images_list=args.save_dir + '/' + 'test.txt', labels_list=args.save_dir + '/' + 'test_labels.txt', resample=args.resample,
                          new_resolution=args.new_resolution, patch_size_x=args.patch_size[0], patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],
                          stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1)

    check_accuracy_model(model, images_list=args.save_dir + '/' + 'train.txt', labels_list=args.save_dir + '/' + 'train_labels.txt',  resample=args.resample,
                         new_resolution=args.new_resolution, patch_size_x=args.patch_size[0], patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],
                         stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1)

end = time()
print('Elapsed time:', round((end - start)/60, 2))

