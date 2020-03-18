"""
@brief  PyTorch inference code for segmentation.

        A 3D fetal brain MRI is segmented using a pre-trained CNN.
        White matter, Ventricles and Cerebellum are segmented.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   March 2020.
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import nibabel as nib
from unet import UNet3D
from datasets.inference_single_volume_dataset import SingleVolumeDataset
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run inference for segmentation')
# Data options
parser.add_argument('--input', default='.', type=str, help='Path of the 3D MRI to segment.')
parser.add_argument('--output_folder', default='', type=str,
                    help='(optional) Path of the folder where to store the segmentations.'
                         'By default the folder containing the input image is used.')
# Model options
parser.add_argument(
    '--model',
    default='./model/3dunet_model.pt7',
    type=str,
    help='Path to the parameters of the pre-trained model.',
)
parser.add_argument('--patch_size', default=136, type=int)

VERSION='0.1.0'

# Hyperparameters
NUM_CHANNELS = 1
NUM_CLASSES = 4


def create_dataset(opt):
    """
    Create the dataset.
    In PyTorch the dataset is responsible for loading the data,
    pre-processing them, and apply data augmentation (optional)
    :param opt: dict of parsed command line arguments
    :return: PyTorch dataset
    """
    # Note that the data are already normalised for segmentation.
    # Prepare normalization and data augmentation layers.
    # Transformations for the input images that are applied before giving them
    # to the network.
    transform = None
    # Create the dataset.
    dataset = SingleVolumeDataset(
        img_path=opt.input,
        path_size=opt.patch_size,
        transform=transform,
    )
    return dataset

def get_network(num_channels, num_classes):
    """
    Return the deep neural network architecture.
    :param num_channels: number of input channels.
    :return: neural network model.
    """
    # Typical Unet architecture used in SoA pipeline
    network = UNet3D(
        in_channels=num_channels,
        out_classes=num_classes,
        out_channels_first_layer=30,
        residual=False,
        normalization='instance',
        padding=True,
        activation='LeakyReLU',
        upsampling_type='trilinear',
    )
    # Set the network in evaluation mode
    network.eval()
    # Put the model on the gpu.
    if torch.cuda.is_available():
        network.cuda()
    return network

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(
            params.cuda() if torch.cuda.is_available() else params,
            dtype)()

def main(opt):
    """
    Run inference for a network that have been trained with main_seg.py.
    :param opt: parsed command line arguments.
    """

    def create_iterator():
        return DataLoader(
            create_dataset(opt),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            # pin_memory=torch.cuda.is_available(),
            pin_memory=False,
        )

    def infer(sample):
        """
        :param sample: couple of tensors; input batch and
        corresponding ground-truth segmentations.
        :return: float, 1d tensor; mean loss for the input batch
        and batch of predicted segmentations.
        """
        with torch.no_grad():  # Tell PyTorch to not store data for backpropagation
            y = network(cast(sample[0], 'float'))
        return y

    def postprocess_and_save(output):
        # Predicted segmentation
        pred = torch.argmax(output.cpu().detach(), dim=1, keepdim=True)

        # Remove dimension of size 1
        pred_numpy = np.squeeze(pred.numpy()).astype(np.int32)

        # Get info about the original image and the patch
        input_shape = img_loader.dataset.input_shape
        coord_min = img_loader.dataset.coord_min
        coord_max = img_loader.dataset.coord_max
        affine = img_loader.dataset.input_affine
        header = img_loader.dataset.input_header

        # Add version to the header (Test)
        header['aux_file'] = 'FetalBrainParcellation-v%s' % VERSION

        # Create the parcellation
        pred_ori_size = np.zeros(input_shape).astype(int)
        pred_ori_size[coord_min[0]:coord_max[0], coord_min[1]:coord_max[1], coord_min[2]:coord_max[2]] = pred_numpy

        # Define the folder where to save the segmentation
        if opt.output_folder == '':
            # By default the segmentations are saved in the folder of the input.
            save_folder = os.path.split(opt.input)[0]
        else:
            save_folder = opt.output_folder
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

        # Get the name of the input
        name = img_loader.dataset.current_sample_id

        # Save the predicted parcellation
        save_path = os.path.join(save_folder, '%s_parcellation_autoseg.nii.gz' % name)
        pred_nii = nib.Nifti1Image(pred_ori_size, affine, header)
        nib.save(pred_nii, save_path)

        # Save independent segmentations
        # WM
        pred_wm_numpy = np.zeros_like(pred_ori_size).astype(int)
        pred_wm_numpy[pred_ori_size == 1] = 1
        pred_wm_nii = nib.Nifti1Image(pred_wm_numpy, affine, header)
        save_path_wm = os.path.join(save_folder, '%s_wm_autoseg.nii.gz' % name)
        nib.save(pred_wm_nii, save_path_wm)
        # CSF (Ventricles)
        pred_csf_numpy = np.zeros_like(pred_ori_size).astype(int)
        pred_csf_numpy[pred_ori_size == 2] = 1
        pred_csf_nii = nib.Nifti1Image(pred_csf_numpy, affine, header)
        save_path_csf = os.path.join(save_folder, '%s_csf_autoseg.nii.gz' % name)
        nib.save(pred_csf_nii, save_path_csf)
        # Cerebellum
        pred_cerebellum_numpy = np.zeros_like(pred_ori_size).astype(int)
        pred_cerebellum_numpy[pred_ori_size == 3] = 1
        pred_cerebellum_nii = nib.Nifti1Image(pred_cerebellum_numpy, affine, header)
        save_path_cerebellum = os.path.join(save_folder, '%s_cerebellum_autoseg.nii.gz' % name)
        nib.save(pred_cerebellum_nii, save_path_cerebellum)

    def restore(model_path):
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            print('Load the model on the CPU...')
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        network.load_state_dict(state_dict['params'])

    # Create the image loader
    img_loader = create_iterator()

    # Create the network
    network = get_network(NUM_CHANNELS, NUM_CLASSES)

    # Initialize the network
    model_path = opt.model
    assert os.path.exists(
        model_path), "Cannot find the model %s" % model_path
    restore(model_path)

    # Run inference for all samples and save output seg
    # Only the central patch at the moment
    for sample in tqdm(img_loader, dynamic_ncols=True):
        output = infer(sample)
        postprocess_and_save(output)


if __name__ == '__main__':
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    main(opt)