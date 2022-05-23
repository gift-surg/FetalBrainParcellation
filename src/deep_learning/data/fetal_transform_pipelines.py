# Copyright 2022 Lucas Fidon
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CastToTyped,
    ToTensord,
)


def fetal_inference_transform(config, image_keys):
    # NB: do not pad; this is done in the segment function for inference
    inference_transform = Compose([
        LoadImaged(keys=image_keys),
        EnsureChannelFirstd(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        Spacingd(
            keys=image_keys,
            pixdim=config['data']['spacing'],
            mode="bilinear",
        ),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=image_keys, dtype=(np.float32,)),
        ToTensord(keys=image_keys),
    ])
    return inference_transform
