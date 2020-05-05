from utils.generator import *
from models_networks.unet3d import *
from training_unet3d import *
from predict import *
import argparse


'''Check if the images and the labels have different size after resampling (or not) them to the same resolution'''

parser = argparse.ArgumentParser()
parser.add_argument("--images_folder", type=str, default='./Data_folder/volumes', help='path to the .nii images')
parser.add_argument("--labels_folder", type=str, default='./Data_folder/labels', help='path to the .nii labels')
parser.add_argument("--resample", action='store_true', default=True, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(0.9375, 0.9375, 3.0), help='New resolution')
args = parser.parse_args()

images = lstFiles(args.images_folder)
labels = lstFiles(args.labels_folder)

for i in range(len(images)):

    a = sitk.ReadImage(images[i])
    if args.resample is True:
        a = resample_sitk_image(a, spacing=args.new_resolution, interpolator='linear')
    spacing1 = a.GetSpacing()
    a = sitk.GetArrayFromImage(a)
    a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
    a1 = a.shape

    b = sitk.ReadImage(labels[i])
    if args.resample is True:
        b = resample_sitk_image(b, spacing=args.new_resolution, interpolator='nearest')
    spacing2 = b.GetSpacing()
    b = sitk.GetArrayFromImage(b)
    b = np.transpose(b, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
    b1 = b.shape

    print(a1)

    if a1 != b1:
        print('Mismatch of size in ', images[i])

# -----------------






































# a=sitk.ReadImage('aaaaaa.nii')
# a = sitk.GetArrayFromImage(a)
# a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
# result = np.rot90(a, k=-1)
# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, result)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()

# a=sitk.ReadImage(labels[36])
# a = sitk.GetArrayFromImage(a)
# a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
# result = np.rot90(a, k=-1)
# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, result)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()



