# Status update - 2023-06-22

⚠️ **This repository is not actively maintained anymore**. Maintenance has moved to [LucasFidon/trustworthy-ai-fetal-brain-segmentation]([https://github.com/Project-MONAI/MONAI/](https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation)).

# Trustworthy Fetal Brain 3D T2w MRI Segmentation
A tool for the automatic segmentation of fetal brain 3D T2-weighted MRI.

![auto-seg](https://user-images.githubusercontent.com/17875992/174453165-2ab9c26b-14da-4728-bec3-710166d12f7b.gif)

## System requirements
### Hardware requirements
To run the automatic segmentation algorithms a NVIDIA GPU with at least 8GB of memory is required.

The code has been tested with the configuration:
* 12 Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
* 1 NVIDIA GPU GeForce GTX 1070 with 8GB of memory

### Operating system requirements
The code is supported on every operating system (OS) using docker.
It has been tested for
* Linux Ubuntu 18.04.6 LTS


## Installation
The installation is performed using docker:

First, install docker (see https://docs.docker.com/get-docker/).

### Download the docker image
Download the latest docker image using
```bash
docker pull lucasfidon/fetal_brain_segmentation:0.1 
```

## How to use

### Automatic Fetal Brain 3D MRI Segmentation
You can compute the automatic segmentations for a fetal brain 3D T2w MRI using the docker image downloaded above.


To learn more about the usage of the docker image, please see
```bash
docker run --rm lucasfidon/fetal_brain_segmentation:0.1 -h
```

We refer to the demo below for a detailed example.

### Demonstration: example case
Fetal brain 3D MRI from a subset of the training dataset can be downloaded at
https://zenodo.org/record/6405632#.YkbWPCTMI5k

Unzip the archive ```FeTA2021_Release1and2Corrected_v2.zip```

Go to the folder containing the 3D MRI examples
```bash
cd FeTA2021_Release1and2Corrected_v2
``` 

We will segment the 3D T2w MRI contained in the folder ```sub-041```.
The folder contains:
* ```srr.nii.gz```: the 3D T2w MRI to segment. This is the main input of the segmentation algorithm.
* ```mask.nii.gz```: the brain mask for ```srr.nii.gz```. This is the second input of the segmentation algorithm.
* ```parcellation.nii.gz```: the manual segmentation for ```srr.nii.gz```. After computing the automatic segmentation you can compare it to this segmentation.

Create a folder for the results of the automatic segmentation algorithm for the case ```sub-041```
```bash
mkdir results-sub-041
```

Run the automatic segmentation for the case ```sub-041/srr.nii.gz``` using
```bash
docker run -v <absolute-path-to-sub-041>:/input -v <absolute-path-to-results-sub-041>:/output --gpus 0 --rm lucasfidon/fetal_brain_segmentation:0.1 --img /input/srr.nii.gz --mask /input/mask.nii.gz --output_folder /output
```
This will take between 1 minute and 3 minutes.

The automatic segmentation will be in saved in ```results-sub-041/srr_segmentation_AI.nii.gz```.


## How to cite
If you find this repository useful for your research, please consider giving us a star :star: and cite
* L. Fidon, M. Aertsen, F. Kofler, A. Bink, A. L. David, T. Deprest, D. Emam, F. Guffens, A. Jakab, G. Kasprian,
 P. Kienast, A. Melbourne, B. Menze, N. Mufti, I. Pogledic, D. Prayer, M. Stuempflen, E. Van Elslander, S. Ourselin, 
 J. Deprest, T. Vercauteren.
 [A Dempster-Shafer approach to trustworthy AI with application to fetal brain MRI segmentation][twai]

```
@article{fidon2022dempster,
  title={A Dempster-Shafer approach to trustworthy AI with application to fetal brain MRI segmentation},
  author={Fidon, Lucas and Aertsen, Michael and Kofler, Florian and Bink, Andrea and David, Anna L and Deprest, Thomas and Emam, Doaa and Guffens, Fr and Jakab, Andr{\'a}s and Kasprian, Gregor and others},
  journal={arXiv preprint arXiv:2204.02779},
  year={2022}
}
```

[twai]: https://arxiv.org/abs/2204.02779
