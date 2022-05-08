# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset, DataLoader, NiftiSaver
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from src.deep_learning.networks.factory import get_network
from src.deep_learning.data.factory import get_single_case_dataloader
from data.config.loader import load_config
from data.dataset_config.loader import load_feta_data_config
from src.utils.definitions import \
    MODELS_PATH, NUM_ITER_MASK_DILATION_BEFORE_INFERENCE, LABELS, CHALLENGE_LABELS


def _check_input_path(data_config, input_path_dict):
    for key in data_config['info']['image_keys']:
        assert key in list(input_path_dict.keys()), 'Input key %s not found in the input paths provided' % key


def pred_softmax(img_path, mask_path, save_folder, convert_labels=True):
    # Load the config files
    config = load_config()
    data_config = load_feta_data_config()

    # Deep learning preprocessing
    preprocessed_img_path = _preprocessing(
        img_path=img_path,
        mask_path=mask_path,
        save_folder=save_folder,
    )

    # Run the inference
    input_path_dict = {'srr': preprocessed_img_path}
    mean_softmax = 0.
    meta_data = None
    for model_path in MODELS_PATH:
        print('Run the inference for %s' % model_path)
        new_pred, meta_data = _pred_softmax_one_model(
            config=config,
            data_config=data_config,
            model_path=model_path,
            input_path_dict=input_path_dict,
        )
        mean_softmax += new_pred
    mean_softmax /= len(MODELS_PATH)

    # Post-processing
    if convert_labels:
        mean_softmax = _postprocessing(mean_softmax=mean_softmax)

    # Save the softmax segmentation
    saver = NiftiSaver(output_dir=save_folder, output_postfix="softmax")
    saver.save_batch(mean_softmax, meta_data=meta_data)

    # Save the segmentation (for inspection)
    seg = mean_softmax.argmax(dim=1, keepdims=True).float()
    saver_seg = NiftiSaver(output_dir=save_folder, output_postfix="seg")
    saver_seg.save_batch(seg, meta_data=meta_data)


def _preprocessing(img_path, mask_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    img_nii = nib.load(img_path)
    img_np = img_nii.get_fdata().astype(np.float32)
    mask_nii = nib.load(mask_path)
    mask_np = mask_nii.get_fdata().astype(np.uint8)

    # Mask the Nans
    if np.count_nonzero(np.isnan(mask_np)) > 0:
        mask_np[np.isnan(mask_np)] = 0

    # Dilate the mask
    mask_dilated_np = binary_dilation(
        mask_np, iterations=NUM_ITER_MASK_DILATION_BEFORE_INFERENCE)

    # Mask the image
    img_np[mask_dilated_np == 0] = 0

    # Clip high intensities
    p_999 = np.percentile(img_np, 99.9)
    img_np[img_np > p_999] = p_999

    # Save the preprocessed image
    new_img_nii = nib.Nifti1Image(img_np, img_nii.affine)
    save_path = os.path.join(save_folder, 'srr_preprocessed.nii.gz')
    nib.save(new_img_nii, save_path)

    return save_path


def _postprocessing(mean_softmax):
    res = torch.zeros_like(mean_softmax)

    # Change the order of the classes to match the convention of the challenge
    for roi in list(LABELS.keys()):
        res[:, CHALLENGE_LABELS[roi], ...] = mean_softmax[:, LABELS[roi], ...]

    return res


def _pred_softmax_one_model(config, data_config, model_path, input_path_dict):
    def pad_if_needed(img, patch_size):
        # Define my own dummy padding function because the one from MONAI
        # does not retain the padding values, and as a result
        # we cannot unpad after inference...
        img_np = img.cpu().numpy()
        shape = img.shape[2:]
        need_padding = np.any(shape < np.array(patch_size))
        if not need_padding:
            pad_list = [(0, 0)] * 3
            return img, np.array(pad_list)
        else:
            pad_list = []
            for dim in range(3):
                diff = patch_size[dim] - shape[dim]
                if diff > 0:
                    margin = diff // 2
                    pad_dim = (margin, diff - margin)
                    pad_list.append(pad_dim)
                else:
                    pad_list.append((0, 0))
            padded_array = np.pad(
                img_np,
                [(0, 0), (0, 0)] + pad_list,  # pad only the spatial dimensions
                'constant',
                constant_values=[(0, 0)] * 5,
            )
            padded_img = torch.tensor(padded_array).float()
            return padded_img, np.array(pad_list)

    # Check that the provided input paths and the data config correspond
    _check_input_path(data_config, input_path_dict)

    device = torch.device("cuda:0")

    # Create the dataloader for the single case to segment
    dataloader = get_single_case_dataloader(
        config=config,
        data_config=data_config,
        input_path_dict=input_path_dict,
    )

    # Create the network and load the checkpoint
    net = get_network(
        config=config,
        in_channels=data_config['info']['in_channels'],
        n_class=data_config['info']['n_class'],
        device=device,
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    # The inferer is in charge of taking a full volumetric input
    # and run the window-based prediction using the network.
    inferer = SlidingWindowInferer(
        roi_size=config['data']['patch_size'],  # patch size to use for inference
        sw_batch_size=1,  # max number of windows per network inference iteration
        overlap=0.5,  # amount of overlap between windows (in [0, 1])
        mode="gaussian",  # how to blend output of overlapping windows
        sigma_scale=0.125,  # sigma for the Gaussian blending. MONAI default=0.125
        padding_mode="constant",  # for when ``roi_size`` is larger than inputs
        cval=0.,  # fill value to use for padding
    )

    torch.cuda.empty_cache()

    net.eval()  # Put the CNN in evaluation mode
    with torch.no_grad():  # we do not need to compute the gradient during inference
        # Load and prepare the full image
        data = [d for d in dataloader][0]  # load the full image
        input = torch.cat(tuple([data[key] for key in data_config['info']['image_keys']]), 1)
        input, pad_values = pad_if_needed(input, config['data']['patch_size'])
        input = input.to(device)
        pred = inferer(inputs=input, network=net)
        n_pred = 1
        # Perform test-time flipping augmentation
        flip_dims = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
        for dims in flip_dims:
            flip_input = torch.flip(input, dims=dims)
            pred += torch.flip(
                inferer(inputs=flip_input, network=net),
                dims=dims,
            )
            n_pred += 1
        pred /= n_pred

    # Unpad the score prediction
    score = pred[:, :, pad_values[0,0]:pred.size(2)-pad_values[0,1], pad_values[1,0]:pred.size(3)-pad_values[1,1], pad_values[2,0]:pred.size(4)-pad_values[2,1]]
    softmax = F.softmax(score, dim=1)

    # Insert the softmax segmentation in the original image size
    meta_data = data['%s_meta_dict' % data_config['info']['image_keys'][0]]
    dim = meta_data['spatial_shape'].cpu().numpy()
    full_dim = [softmax.size(0), softmax.size(1), dim[0,0], dim[0,1], dim[0,2]]
    fg_start = data['foreground_start_coord'][0]
    fg_end = data['foreground_end_coord'][0]
    full_softmax = torch.zeros(full_dim)
    full_softmax[:, 0, ...] = 1.  # background by default
    full_softmax[:, :, fg_start[0]:fg_end[0], fg_start[1]:fg_end[1], fg_start[2]:fg_end[2]] = softmax

    return full_softmax, meta_data
