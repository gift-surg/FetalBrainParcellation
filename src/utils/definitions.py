import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

HOME_FOLDER = '/'
# WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'feta_seg')
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace', 'feta-inference')
DATA_FOLDER = os.path.join(WORKSPACE_FOLDER, 'data')
CNN_WEIGHTS_FOLDER = os.path.join(DATA_FOLDER, 'trained_models')
MODELS_PATH = [
    os.path.join(
        CNN_WEIGHTS_FOLDER,
        'feta_split%d' %i,
        'checkpoint_epoch=2200.pt',
    )
    for i in range(10)
]

# ATLAS FOLDERS
DATA_FOLDER_HARVARD_GROUP = os.path.join(
    DATA_FOLDER,
    'fetal_brain_atlases',
    'Gholipour2017_atlas_NiftyMIC_preprocessed_corrected',
)
DATA_FOLDER_SPINA_BIFIDA_ATLAS = os.path.join(
    DATA_FOLDER,
    'fetal_brain_atlases',
    'spina_bifida_atlas',
)

LABELS = {
    'wm': 1,
    'csf': 2,  # ventricles
    'cerebellum': 3,
    'external_csf': 4,
    'cortical_gm': 5,
    'deep_gm': 6,
    'brainstem': 7,
    'background': 0,
}

CHALLENGE_LABELS = {
    'wm': 3,
    'csf': 4,  # ventricles
    'cerebellum': 5,
    'external_csf': 1,
    'cortical_gm': 2,
    'deep_gm': 6,
    'brainstem': 7,
    'background': 0,
}

# HYPER-PARAMETERS
NUM_ITER_MASK_DILATION_BEFORE_INFERENCE = 5