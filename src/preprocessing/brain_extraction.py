"""
Atlas-based brain extraction.
Compute a brain mask for the SRR.

@author: Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from argparse import ArgumentParser
import numpy as np
import nibabel as nib
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.utils.definitions import DATA_FOLDER_HARVARD_GROUP, DATA_FOLDER_SPINA_BIFIDA_ATLAS


N_ITER_DILATION_MASK = 1
INT_THRES_INIT_MASK = 1.  # Intensity threshold used to initialize the brain mask
CONDITIONS = ['Neurotypical', 'Pathological']


parser = ArgumentParser()
parser.add_argument('--input_img', required=True,
                    help='Path to the SRR to preprocess')
parser.add_argument('--output_folder',
                    help='Folder where to save the mask.nii.gz file. '
                         'By default the folder of the input image is used.')
parser.add_argument('--ga', required=True, type=int, help='gestational age')
parser.add_argument('--spina_bifida', action='store_true')
parser.add_argument('--no_threshold_initialization', action='store_true')


def get_template_path(gestational_age, condition='Neurotypical'):
    if condition == 'Neurotypical':
        folder = os.path.join(
            DATA_FOLDER_HARVARD_GROUP,
            "HarvardSTA%d_Study1" % gestational_age
        )
    else:
        folder = os.path.join(
            DATA_FOLDER_SPINA_BIFIDA_ATLAS,
            'fetal_SB_atlas_GA%d_notoperated' % gestational_age
        )
        if not os.path.exists(folder):
            folder = os.path.join(
                DATA_FOLDER_SPINA_BIFIDA_ATLAS,
                'fetal_SB_atlas_GA%d_operated' % gestational_age
            )
    template = os.path.join(folder, "srr.nii.gz")
    template_mask = os.path.join(folder, "mask.nii.gz")
    return template, template_mask


def create_mask_thresholding(img_path, save_mask_path, threshold=0):
    """
    Create an initial mask using image intensity thresholding.
    :param img_path:
    :param save_mask_path:
    :param threshold:
    :return:
    """
    img_nii = nib.load(img_path)
    img_np = img_nii.get_fdata().astype(np.float32)
    # Remove NaNs if needed
    num_nans = np.count_nonzero(np.isnan(img_np))
    if num_nans > 0:
        img_np[np.isnan(img_np)] = threshold - 1
    # Compute and save the mask obtained by thresholding
    mask_np = (img_np > threshold).astype(np.uint8)
    mask_nii = nib.Nifti1Image(mask_np, img_nii.affine)
    nib.save(mask_nii, save_mask_path)


def get_brain_mask(path_srr, gestational_age, dir_output, tmp_folder,
                   path_initial_mask=None, rig_only=False, spina_bifida=False):
    """
    An estimation of the brain mask is obtained by registering the templates
    into the input image, and propagating the mask of the template.
    """
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    out_mask_path = os.path.join(dir_output, 'mask.nii.gz')
    out_proba_mask_path_list = []

    # We register the several templates volumes to the target SRR
    print('Use the template volumes:')
    for ga_delta in [-1, 0, 1]:
        ga = gestational_age + ga_delta
        if ga < 21:
            continue
        if ga > 38:
            continue
        if spina_bifida:
            condition_list = ['Pathological']
        else:
            condition_list = CONDITIONS
        for condition in condition_list:
            if condition != 'Neurotypical' and ga > 34:
                continue
            template, template_mask = get_template_path(ga, condition)
            print(os.path.split(os.path.dirname(template))[1])

            template_warp_aff = os.path.join(
                tmp_folder, 'template_warp_aff_GA%d_%s.nii.gz' % (ga, condition))
            save_aff = os.path.join(tmp_folder, 'affine_GA%d_%s.txt' % (ga, condition))
            cmd = 'reg_aladin '
            cmd += '-ref %s ' % path_srr
            cmd += '-fmask %s ' % template_mask
            if path_initial_mask is not None:
                cmd += '-rmask %s ' % path_initial_mask
                cmd += '-comm '  # use the input masks centre of mass to initialise the transformation
            if rig_only:
                cmd += '-rigOnly'
            cmd += '-flo %s ' % template
            cmd += '-res %s ' % template_warp_aff
            cmd += '-aff %s ' % save_aff
            cmd += '-voff'
            os.system(cmd)

            template_mask_warp_aff = os.path.join(
                tmp_folder, 'template_mask_warp_GA%d_%s.nii.gz' % (ga, condition))
            cmd = 'reg_resample '
            cmd += '-ref %s ' % path_srr
            cmd += '-flo %s ' % template_mask
            cmd += '-trans %s ' % save_aff
            cmd += '-res %s ' % template_mask_warp_aff
            cmd += '-inter 0 -voff'
            os.system(cmd)

            out_proba_mask_path_list.append(template_mask_warp_aff)

    # Compute the average mask
    proba_mask_np = 0
    aff = None
    print('%d masks are averaged' % len(out_proba_mask_path_list))
    for mask_path in out_proba_mask_path_list:
        mask_nii = nib.load(mask_path)
        proba_mask_np += mask_nii.get_fdata().astype(np.float32)
        if aff is None:
            aff = mask_nii.affine
    proba_mask_np /= len(out_proba_mask_path_list)
    mask_np = (proba_mask_np > 0.5).astype(np.uint8)

    # Post-processing
    if N_ITER_DILATION_MASK > 0:
        print('Dilate the estimated brain mask with %d iterations' % N_ITER_DILATION_MASK)
        mask_np = binary_dilation(
            mask_np,
            iterations=N_ITER_DILATION_MASK).astype(np.uint8)
    # Fill holes
    mask_np = binary_fill_holes(mask_np).astype(np.uint8)

    # Save the output mask
    mask_nii = nib.Nifti1Image(mask_np, aff)
    nib.save(mask_nii, out_mask_path)

    return out_mask_path


def main(args):
    if args.output_folder is None:
        out_folder = os.path.dirname(args.input_img)
    else:
        out_folder = args.output_folder

    tmp_folder = os.path.join(out_folder, 'brain_extraction')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    if args.no_threshold_initialization:
        thres_mask_path = None
    else:
        # Estimate the initial mask using thresholding
        # It will be usually an over-estimation
        print('Create initial mask with threshold equal to 1')
        thres_mask_path = os.path.join(tmp_folder, 'mask_threshold.nii.gz')
        create_mask_thresholding(args.input_img, thres_mask_path, threshold=INT_THRES_INIT_MASK)

    # Estimate the brain mask for the input image in the template space
    print('\nEstimate the brain mask')
    out_mask_path = get_brain_mask(
        path_srr=args.input_img,
        gestational_age=args.ga,
        dir_output=out_folder,
        path_initial_mask=thres_mask_path,
        tmp_folder=tmp_folder,
        spina_bifida=args.spina_bifida,
    )
    print('Brain mask saved at %s' % out_mask_path)

    # if os.path.exists(tmp_folder):
    #     os.system('rm -r %s' % tmp_folder)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
