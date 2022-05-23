# Copyright 2022 Lucas Fidon
import os
import numpy as np
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

HOME_FOLDER = '/'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace', 'FetalBrainParcellation')
DATA_FOLDER = os.path.join(WORKSPACE_FOLDER, 'data')
CNN_WEIGHTS_FOLDER = os.path.join(DATA_FOLDER, 'trained_deep_learning_models')
MODELS_PATH = [
    os.path.join(
        CNN_WEIGHTS_FOLDER,
        'model_apr22_split%d.pt' %i,
    )
    for i in range(6)
]

# ATLAS FOLDERS
ATLAS_CONTROL_HARVARD = os.path.join(  # GA: 21 -> 37
    DATA_FOLDER,
    'fetal_brain_atlases',
    'Neurotypical_Gholipour2017',
)
ATLAS_CONTROL_CHINESE = os.path.join(  # GA: 22 -> 35
    DATA_FOLDER,
    'fetal_brain_atlases',
    'Neurotypical_Wu2021',
)
ATLAS_SB = os.path.join(
    DATA_FOLDER,
    'fetal_brain_atlases',
    'SpinaBifida_Fidon2021',
)


# SEGMENTATION PARAMETERS
LABELS = {
    'wm': 1,
    'intra_csf': 2,
    'cerebellum': 3,
    'external_csf': 4,
    'cortical_gm': 5,
    'deep_gm': 6,
    'brainstem': 7,
    'cc': 8,
    'background': 0,
}
NUM_ITER_MASK_DILATION_BEFORE_INFERENCE = 5

# REGISTRATION HYPER-PARAMETERS
IMG_RES = 0.8  # in mm; isotropic
CONDITIONS = ['Neurotypical', 'Spina Bifida', 'Pathological']
GRID_SPACING = 4  # in mm (default is 4 mm = 5 voxels x 0.8 mm.voxels**(-1))
BE = 0.1
LE = 0.3
LP = 3  # default 3; we do only the lp first level of the pyramid
DELTA_GA_CONTROL = 1
DELTA_GA_SPINA_BIFIDA = 3
ATLAS_MARGINS_CONTROL_MM = np.array([1.6, 1.6, 1.1, 1.6, 0.8, 1.4, 2.4, 2.9, 1.1])
ATLAS_MARGINS_CONTROL = ATLAS_MARGINS_CONTROL_MM / IMG_RES
ATLAS_MARGINS_SPINA_BIFIDA_MM = np.array([1.6, 2.0, 1.0, 2.7, 2.3, 2.0, 1.8, 3.4, 1.7])
ATLAS_MARGINS_SPINA_BIFIDA = ATLAS_MARGINS_SPINA_BIFIDA_MM / IMG_RES
MIN_GA = 21
MAX_GA = 38
MERGING_MULTI_ATLAS = 'GIF'
