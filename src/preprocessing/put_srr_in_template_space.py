"""
Use this script to pre-process external data (not reconstructed using NiftyMIC)
This script performs rigid registration of the SRR to the template space.

@author: Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
from argparse import ArgumentParser
import numpy as np
import nibabel as nib
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.preprocessing.brain_extraction import get_template_path


parser = ArgumentParser()
parser.add_argument('--input_img', required=True,
                    help='Path to the SRR to preprocess')
parser.add_argument('--input_mask', required=True,
                    help='Path to the brain mask (usually obtained with brain_extraction.py)')
parser.add_argument('--output_folder', required=True)
parser.add_argument('--ga', required=True, type=int, help='gestational age')
parser.add_argument('--spina_bifida', action='store_true')


def register_to_template_space(path_srr, gestational_age, dir_output,
                               tmp_folder, path_mask, spina_bifida=False):
    """
    Register the SRR in path_srr to the normal fetal brain atlas with the same GA.
    A rigid transformation is used.
    The warped SRR and the rigid transformation are saved in dir_output.
    """
    condition = 'Pathological' if spina_bifida else 'Neurotypical'
    template, template_mask = get_template_path(gestational_age, condition=condition)
    print(template)
    save_path = os.path.join(
        dir_output,
        'srr.nii.gz'
    )
    affine_path = os.path.join(tmp_folder, 'affine.txt')
    cmd = 'reg_aladin '
    cmd += '-ref %s ' % template
    cmd += '-rmask %s ' % template_mask
    cmd += '-flo %s ' % path_srr
    cmd += '-fmask %s ' % path_mask
    cmd += '-res %s ' % save_path
    cmd += '-aff %s ' % affine_path
    cmd += '-comm '  # use the input masks centre of mass to initialise the transformation
    cmd += '-pad 0 '
    cmd += '-rigOnly -voff'
    os.system(cmd)
    return save_path, affine_path


def warp_seg(seg_path, ref_img_path, save_path, affine_transform_path, tmp_folder):
    """
    Apply an affine transformation to a segmentation.

    For the registration, the segmentation is converted into a one-hot representation
    and a linear interpolation is used.

    This is found to lead to more accurate segmentation registration as compared
    to nearest neighbor registration.
    """
    def _convert_to_one_hot_and_smooth_seg_prior(segmentation_nii):
        seg_np = segmentation_nii.get_fdata().astype(np.uint8)
        # Convert the segmentation into one-hot representation
        seg_one_hot = np.eye(seg_np.max() + 1)[seg_np].astype(np.float32)  # numpy magic
        one_hot_nii = nib.Nifti1Image(
            seg_one_hot,
            segmentation_nii.affine,
        )
        return one_hot_nii

    # Convert the segmentation into one hot representation
    seg_nii = nib.load(seg_path)
    seg_onehot_nii = _convert_to_one_hot_and_smooth_seg_prior(seg_nii)
    seg_onehot_path = os.path.join(tmp_folder, 'seg_onehot.nii.gz')
    nib.save(seg_onehot_nii, seg_onehot_path)

    # Affine deformation of the one hot segmentation
    warped_onehot_seg = os.path.join(tmp_folder, 'warped_seg_onehot.nii.gz')
    cmd = 'reg_resample -ref %s -flo %s -trans %s -res %s -inter 1 -pad 0 -voff' % \
        (ref_img_path, seg_onehot_path, affine_transform_path, warped_onehot_seg)
    os.system(cmd)

    # Convert the warped one hot segmentation into a normal segmentation
    warped_onehot_seg_nii = nib.load(warped_onehot_seg)
    warped_onehot_seg_np = warped_onehot_seg_nii.get_fdata()
    warped_seg_np = np.argmax(warped_onehot_seg_np, axis=-1).astype(np.uint8)
    warped_seg_nii = nib.Nifti1Image(warped_seg_np, warped_onehot_seg_nii.affine)
    nib.save(warped_seg_nii, save_path)


def main(args):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    tmp_folder = os.path.join(args.output_folder, 'tmp_put_srr_to_template_space')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    img_path = args.input_img
    mask_path = args.input_mask

    # Register the input image and the seg into the template space
    print('\nRegister the input SRR to the template space')
    srr_template_space_path, affine_path = register_to_template_space(
        path_srr=img_path,
        gestational_age=args.ga,
        dir_output=args.output_folder,
        path_mask=mask_path,
        tmp_folder=tmp_folder,
        spina_bifida=args.spina_bifida,
    )

    # Warp the mask
    mask_template_space_path = os.path.join(args.output_folder, 'mask.nii.gz')
    warp_seg(
        seg_path=mask_path,
        ref_img_path=srr_template_space_path,
        save_path=mask_template_space_path,
        affine_transform_path=affine_path,
        tmp_folder=tmp_folder,
    )

    # if os.path.exists(tmp_folder):
    #     os.system('rm -r %s' % tmp_folder)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
