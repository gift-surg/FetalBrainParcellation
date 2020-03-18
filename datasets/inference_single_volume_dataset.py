"""
@brief  Pytorch Dataset class used for inference with 3d image segmentation.
        It is designed for running inference on ONLY ONE image.
        The Volume is normalized to zeros mean and unit variance when loaded.


@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   January 2020.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class SingleVolumeDataset(Dataset):
    def __init__(self, img_path, path_size=None, transform=None):
        """
        Dataset class used for inference with 3d image segmentation.
        It is designed for running inference on ONLY ONE image.
        The Volume is normalized to zeros mean and unit variance when loaded.
        :param img_path: str; path to the image.
        :param path_size: int; size of the patch to use.
        :param transform: PyTorch transform layer to use.
        """
        super(SingleVolumeDataset, self).__init__()

        assert os.path.exists(img_path), "Please verify that the image path %s is correct." % img_path
        self.sample_img_path = img_path

        self.samples_id_list = []  # sample ids to use

        sample_id = os.path.split(img_path)[-1].split('.')[0]
        self.samples_id_list.append(sample_id)

        self.current_img_name = None
        self.patch_size = path_size

        print("Found image %s at %s" % (sample_id, img_path))

    def __getitem__(self, index=0):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (subwindow) image to segment.
        """
        self.current_sample_id = self.samples_id_list[index]
        self.coord_min = None
        self.coord_max = None
        self.input_shape = None
        self.input_header = None
        self.input_affine = None

        # load image
        img_nii = nib.load(self.sample_img_path)
        img = img_nii.get_data().astype(np.float32)
        self.input_affine = img_nii.affine
        self.input_header = img_nii.header

        # extract central patch
        if self.patch_size is not None:
            self.input_shape = img.shape
            center_coord = np.array(img.shape) // 2
            coord_min = center_coord - np.array([self.patch_size // 2]*3)
            coord_max = coord_min + np.array([self.patch_size]*3)
            img = img[coord_min[0]:coord_max[0],coord_min[1]:coord_max[1],coord_min[2]:coord_max[2]]
            # Remember the coordinates of the current patch/subwindow
            self.coord_min = coord_min
            self.coord_max = coord_max

        # add chanel dimension
        img = np.expand_dims(img, axis=0)
        # add batch dimension
        img = np.expand_dims(img, axis=0)

        # clip last percentile
        p999 = np.percentile(img, 99.9)
        img[img > p999] = p999

        # whitening of the image
        assert np.abs(img).sum() != 0, "The image is black. " \
                                       "There must be a problem with " \
                                       "your image %s." % self.sample_img_path
        img = img - np.mean(img)
        img = img / np.std(img)

        # Convert to torch tensor
        img = torch.from_numpy(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        return img

    def __len__(self):
        """
        A SingleVolumeDataset contains only one volume.
        """
        return 1