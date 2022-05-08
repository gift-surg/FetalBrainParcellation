# -*- coding: utf-8 -*-

"""
@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
import time
from argparse import ArgumentParser
import numpy as np
import nibabel as nib
from src.deep_learning.inference.inference import pred_softmax


parser = ArgumentParser()
parser.add_argument('--img', required=True,
                    help='Path to the SRR to preprocess')
parser.add_argument('--mask', required=False,
                    help='Path to the SRR to preprocess')
parser.add_argument('--ga', required=True, help='Gestational age')
parser.add_argument('--output_folder', required=False)
parser.add_argument('--spina_bifida', action='store_true')
parser.add_argument('--convert_labels', action='store_true')

REPO_DIR = os.path.join('/workspace', 'feta-inference')


def main(args):
    t0 = time.time()

    # Get paths and subject name
    T2wImagePath = args.img
    sub = os.path.split(T2wImagePath)[1].split('_')[0].replace('.nii.gz', '') # to split the input directory and to obtain the subject name
    if args.output_folder is None:
        outputDir = os.path.dirname(args.img)
    else:
        outputDir = args.output_folder
    print('Output will be saved in %s' % outputDir)
    outputDirTemplateSpace = os.path.join(outputDir, 'template_space')

    # GA to use
    ga_rounded = max(min(int(round(float(args.ga))), 38), 21)  # clip the GA to [21, 38]
    print('GA used:', ga_rounded)

    # PRE-PROCESSING
    # Try to find an existing brain mask
    maskPath = args.mask
    if maskPath is not None and os.path.exists(maskPath):
        print('\n** Found the brain mask %s' % maskPath)
    else:
        print('\n** Create the brain mask (mask.nii.gz saved in outputDir)')
        cmd_brain_extraction = 'python %s/src/preprocessing/brain_extraction.py --input_img %s --output_folder %s --ga %d' % \
            (REPO_DIR, T2wImagePath, outputDir, ga_rounded)
        if args.spina_bifida:
            cmd_brain_extraction += ' --spina_bifida'
        os.system(cmd_brain_extraction)
        maskPath = os.path.join(outputDir, 'mask.nii.gz')

    print('\n** Put the SRR and mask in the template space')
    cmd_put_in_template_space = 'python %s/src/preprocessing/put_srr_in_template_space.py --input_img %s --input_mask %s --output_folder %s --ga %d' % \
        (REPO_DIR, T2wImagePath, maskPath, outputDirTemplateSpace, ga_rounded)
    os.system(cmd_put_in_template_space)
    T2wImageTemplateSpacePath = os.path.join(outputDirTemplateSpace, 'srr.nii.gz')
    maskTemplateSpacePath = os.path.join(outputDirTemplateSpace, 'mask.nii.gz')
    affinePath = os.path.join(outputDirTemplateSpace, 'tmp_put_srr_to_template_space', 'affine.txt')

    # INFERENCE
    print('\n** Run the deep learning ensemble on the SRR in the template space')
    pred_softmax(
        img_path=T2wImageTemplateSpacePath,
        mask_path=maskTemplateSpacePath,
        save_folder=outputDirTemplateSpace,
        convert_labels=args.convert_labels,
    )
    softmaxTemplateSpace = os.path.join(outputDirTemplateSpace, 'srr_preprocessed', 'srr_preprocessed_softmax.nii.gz')

    # POSTPROCESSING
    print('\n** Warp the segmentation to the subject space')
    cmd_put_in_subject_space = 'python %s/src/postprocessing/put_softmax_in_subject_space.py --softmax %s --aff %s --input_img %s --output_folder %s' % \
        (REPO_DIR, softmaxTemplateSpace, affinePath, T2wImagePath, outputDir)
    os.system(cmd_put_in_subject_space)
    softmaxPath = os.path.join(outputDir, 'softmax.nii.gz')
    softmax_nii = nib.load(softmaxPath)
    softmax = softmax_nii.get_fdata().astype(np.float32)
    seg = np.argmax(softmax, axis=-1).astype(np.uint8)
    seg_nii = nib.Nifti1Image(seg, softmax_nii.affine)
    save_path = os.path.join(outputDir, sub + '_seg_result.nii.gz')
    nib.save(seg_nii, save_path)
    print('\nPredicted segmentation saved at %s' % save_path)

    duration = int(time.time() - t0)
    print('Total time: %dmin%dsec' % (duration // 60, duration % 60))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
