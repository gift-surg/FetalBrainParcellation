# -*- coding: utf-8 -*-
# Copyright 2022 Lucas Fidon

"""
@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
import time
from argparse import ArgumentParser
import numpy as np
import nibabel as nib
from src.deep_learning.inference.inference import pred_softmax
from src.utils.definitions import *
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list
from src.segmentations_fusion.dempster_shaffer import merge_deep_and_atlas_seg


parser = ArgumentParser()
parser.add_argument('--img', required=True,
                    help='Path to the SRR to segment')
parser.add_argument('--mask', required=False,
                    help='Path to the brain mask')
# OPTIONS
parser.add_argument('--output_folder', required=False,
                    help='(optional) Path where to save the output. By default the folder of --img will be used.')
# TWAI
parser.add_argument('--trustworthy', action='store_true',
                    help='Use the trustworthy AI approach.')
parser.add_argument('--ga',
                    help='Gestational age (in weeks). ex: 28.3')
parser.add_argument('--spina_bifida', action='store_true',
                    help='To be used with --trustworthy. Use specific method for spina bifida.')
parser.add_argument('--very_abnormal', action='store_true',
                    help='To be used with --trustworthy. Use specific method very abnormal brains other than spina bifida.')
# Pre/Post-processing
parser.add_argument('--compute_brain_mask', action='store_true')
parser.add_argument('--put_in_atlas_space', action='store_true',
                    help='Put the input 3D MRI in a standard atlas space before segmentation')
parser.add_argument('--output_in_input_space', action='store_true',
                    help='Force to return the segmentation in the original space of the 3D MRI --img')


def _get_atlas_volumes_path_list(condition, ga):
    if condition == 'Pathological':
        atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
        atlas_list += get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
    elif condition == 'Neurotypical':
        atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
    else:
        assert condition == 'Spina Bifida', 'Unknown condition %s' % condition
        atlas_list = get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
    return atlas_list


def main(args):
    # INITIALIZATION
    t0 = time.time()
    # Get paths and subject name
    T2wImagePath = args.img
    sub = os.path.split(T2wImagePath)[1].replace('.nii.gz', '')
    if args.output_folder is None:
        outputDir = os.path.dirname(args.img)
    else:
        outputDir = args.output_folder
    print('Output will be saved in %s' % outputDir)
    # Gestational Age (in weeks)
    if args.ga is not None:
        ga = max(min(int(round(float(args.ga))), 38), 21)  # clip the GA to [21, 38]
    else:
        ga = None
    # Trustworthy AI
    use_TWAI = args.trustworthy
    if use_TWAI:
        assert ga is not None, 'Please indicate the gestational age using --ga to use the --trustworthy option.'
    if args.spina_bifida:
        cond = 'Spina Bifida'
    elif args.very_abnormal:
        cond = 'Pathological'
    else:
        cond = 'Neurotypical'

    # PRE-PROCESSING
    # Try to find an existing brain mask
    maskPath = args.mask
    if maskPath is not None and os.path.exists(maskPath):
        print('\n** Found the brain mask: \n%s' % maskPath)
    else:
        raise NotImplementedError()
        # print('\n** Create the brain mask (mask.nii.gz saved in outputDir)')
        # cmd_brain_extraction = 'python %s/src/preprocessing/brain_extraction.py --input_img %s --output_folder %s --ga %d' % \
        #     (REPO_DIR, T2wImagePath, outputDir, ga_rounded)
        # if args.spina_bifida:
        #     cmd_brain_extraction += ' --spina_bifida'
        # os.system(cmd_brain_extraction)
        # maskPath = os.path.join(outputDir, 'mask.nii.gz')
    # print('\n** Put the SRR and mask in the template space')
    # cmd_put_in_template_space = 'python %s/src/preprocessing/put_srr_in_template_space.py --input_img %s --input_mask %s --output_folder %s --ga %d' % \
    #     (REPO_DIR, T2wImagePath, maskPath, outputDirTemplateSpace, ga_rounded)
    # os.system(cmd_put_in_template_space)
    # T2wImageTemplateSpacePath = os.path.join(outputDirTemplateSpace, 'srr.nii.gz')
    # maskTemplateSpacePath = os.path.join(outputDirTemplateSpace, 'mask.nii.gz')
    # affinePath = os.path.join(outputDirTemplateSpace, 'tmp_put_srr_to_template_space', 'affine.txt')

    # INFERENCE
    print('\n** Run the deep learning ensemble on the SRR in the template space')
    pred_softmax(
        img_path=T2wImagePath,
        mask_path=maskPath,
        save_folder=outputDir,
    )
    tmpOutDir = os.path.join(outputDir, 'srr_preprocessed')
    softmaxPath = os.path.join(tmpOutDir, 'srr_preprocessed_softmax.nii.gz')
    if use_TWAI:
        # Init
        img_nii = nib.load(T2wImagePath)
        # img = img_nii.get_fdata().astype(np.float32)
        mask_nii = nib.load(maskPath)
        # mask = mask_nii.get_fdata().astype(np.uint8)
        # Fallback inference - Propagate the atlas volumes segmentation
        atlas_list = _get_atlas_volumes_path_list(cond, ga)
        print('\nStart atlas propagation using the volumes')
        print(atlas_list)
        atlas_pred_save_folder = os.path.join(tmpOutDir, 'atlas_pred')
        pred_proba_atlas = multi_atlas_segmentation(
            img_nii,
            mask_nii,
            atlas_folder_list=atlas_list,
            grid_spacing=GRID_SPACING,
            be=BE,
            le=LE,
            lp=LP,
            save_folder=atlas_pred_save_folder,
            only_affine=False,
            merging_method=MERGING_MULTI_ATLAS,
            reuse_existing_pred=False,
            force_recompute_heat_kernels=False,
        )
        # Transpose the atlas proba to match PyTorch convention
        # pred_proba_atlas = np.transpose(pred_proba_atlas, axes=(3, 0, 1, 2))
        pred_atlas = np.argmax(pred_proba_atlas, axis=-1).astype(np.uint8)
        softmax_nii = nib.load(softmaxPath)
        softmax = softmax_nii.get_fdata().astype(np.float32)
        # Take a weighted average of the CNN and atlas predicted proba
        pred_proba_trustworthy = len(CNN_WEIGHTS_FOLDER) * softmax + pred_proba_atlas
        pred_proba_trustworthy /= len(CNN_WEIGHTS_FOLDER) + 1
        # Apply Dempster's rule with the atlas prior
        softmax = merge_deep_and_atlas_seg(
            deep_proba=pred_proba_trustworthy,
            atlas_seg=pred_atlas,
            condition=cond,  # Used to know which margins to use
        )
    else:
        softmax_nii = nib.load(softmaxPath)
        softmax = softmax_nii.get_fdata().astype(np.float32)

    # POSTPROCESSING
    # print('\n** Warp the segmentation to the subject space')
    # cmd_put_in_subject_space = 'python %s/src/postprocessing/put_softmax_in_subject_space.py --softmax %s --aff %s --input_img %s --output_folder %s' % \
    #     (REPO_DIR, softmaxTemplateSpace, affinePath, T2wImagePath, outputDir)
    # os.system(cmd_put_in_subject_space)
    # softmaxPath = os.path.join(outputDir, 'softmax.nii.gz')

    seg = np.argmax(softmax, axis=-1).astype(np.uint8)
    seg_nii = nib.Nifti1Image(seg, softmax_nii.affine)
    if use_TWAI:
        save_path = os.path.join(outputDir, sub + '_segmentation_TWAI.nii.gz')
    else:
        save_path = os.path.join(outputDir, sub + '_segmentation_AI.nii.gz')
    nib.save(seg_nii, save_path)
    print('\nPredicted segmentation saved at %s' % save_path)

    # Cleaning
    if os.path.exists(tmpOutDir):
        os.system('rm -r %s' % tmpOutDir)

    duration = int(time.time() - t0)
    print('Total time: %dmin%dsec' % (duration // 60, duration % 60))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
