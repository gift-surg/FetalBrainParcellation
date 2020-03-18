# Fetal Brain Parcellation
A tool for the automatic segmentation of 3D reconstructed fetal brain T2-weighted MRI.

## Docker
In this section we describe how to install and use the application using docker.

#### Installation
Open a terminal, go to the folder containing the project, and run
```
pip install -r requirements.txt
```

#### Use FetalBrainParcellation
You can segment a 3D fetal brain MRI located at /dir-to-my-mri/my-mri-file-name.nii.gz 
by running
```
python run_auto_seg.py --input /input/my-mri-file-name.nii.gz --output_folder /my-output-folder
```

#### Remarks
* For optimal segmentation results, the input image should be skull stripped 
before segmentation.
* For optimal segmentation results, use the NiftyMIC software to obtain 3D reconstructed 
fetal brain MRI https://github.com/gift-surg/NiftyMIC
* The code has been tested on Linux Ubuntu 18.02 with python3.6, a GPU GeForce GTX 1070/PCIe/SSE2 with 8Gb of memory 
and the inference time for a full volume was in average of 1sec.
