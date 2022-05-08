"""
Use this script to post-process the predicted softmax segmentation.
This script performs rigid register of the softmax prediction to the subject space.

@author: Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
from argparse import ArgumentParser
import numpy as np
import nibabel as nib

parser = ArgumentParser()
parser.add_argument('--softmax', required=True,
                    help='path to the softmax prediction in the template space.')
parser.add_argument('--aff', required=True,
                    help='path to the Affine transformation that was used'
                         'to go from subject space to template space.')
parser.add_argument('--input_img', required=True,
                    help='Path to the SRR to preprocess')
parser.add_argument('--output_folder', required=True)


def invert_affine(aff_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    aff_name = os.path.split(aff_path)[1].replace('.txt', '')
    save_inv_aff_path = os.path.join(
        output_dir,
        '%s_inv.txt' % aff_name,
    )
    cmd = 'reg_transform -invAff %s %s' % (aff_path, save_inv_aff_path)
    os.system(cmd)
    return save_inv_aff_path


def warp_softmax(softmax_path, ref_img_path, save_path, aff_path):
    # Warp the softmax
    cmd = 'reg_resample -ref %s -flo %s -trans %s -res %s -inter 1 -pad 0 -voff' % \
        (ref_img_path, softmax_path, aff_path, save_path)
    os.system(cmd)

    # Fix border effects due to padding with 0 AND change order of channels
    softmax_nii = nib.load(save_path)
    softmax = softmax_nii.get_fdata().astype(np.float32)
    sum_proba = np.sum(softmax, axis=-1)
    softmax[:, :, :, 0] += 1. - sum_proba
    post_softmax_nii = nib.Nifti1Image(softmax, softmax_nii.affine)
    nib.save(post_softmax_nii, save_path)


def main(args):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    # Compute the inverse affine transform
    print('Invert %s' % args.aff)
    inv_aff_path = invert_affine(aff_path=args.aff, output_dir=args.output_folder)
    print(inv_aff_path)

    # Warp the softmax
    save_path = os.path.join(args.output_folder, 'softmax.nii.gz')
    print('warp %s' % args.softmax)
    warp_softmax(
        softmax_path=args.softmax,
        ref_img_path=args.input_img,
        save_path=save_path,
        aff_path=inv_aff_path,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
