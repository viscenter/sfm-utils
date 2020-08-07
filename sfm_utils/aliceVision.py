import json
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np

from sfm_utils.sfm import Intrinsic, IntrinsicType, Pose, SfMScene, View

__AV_INTRINSIC_NAME_MAP = {
    IntrinsicType.PINHOLE: 'pinhole',
    IntrinsicType.RADIAL_K3: 'radial3',
    IntrinsicType.BROWN_T2: 'brownt2'
}


def export_alicevision(path: Union[str, bytes, PathLike], sfm: SfMScene):
    """
    Export SfMScene to an AliceVision JSON file

    Note: This is currently untested for actual use in MeshRoom
    """

    def av_view(view: View):
        """
        AliceVision View struct
        """
        d = {
            "viewID": str(view.id),
            "poseID": str(view.pose.id),
            "intrinsicID": str(view.intrinsic.id),
            "path": str(view.path),
            "width": str(view.width),
            "height": str(view.height)
        }
        return d

    def av_intrinsic(intrinsic: Intrinsic):
        """
        AliceVision Intrinsic struct
        """
        d = {
            "intrinsicID": str(intrinsic.id),
            "width": str(intrinsic.width),
            "height": str(intrinsic.height),
            "serialNumber": str(intrinsic.id),
            "type": __AV_INTRINSIC_NAME_MAP[intrinsic.type],
            "initializationMode": "estimated",
            "pxInitialFocalLength": str(intrinsic.focal_length_as_pixels),
            "pxFocalLength": str(intrinsic.focal_length_as_pixels),
            "principalPoint": [
                str(intrinsic.ppx),
                str(intrinsic.ppy)
            ],
            "locked": "0"
        }

        # Add any distortion parameters
        if intrinsic.dist_params is not None:
            d['distortionParams'] = [str(i) for i in intrinsic.dist_params]

        return d

    def av_pose(pose: Pose):
        """
        AliceVision Pose struct
        """
        d = {
            "poseId": str(pose.id),
            "pose": {
                "transform": {
                    "rotation": [str(i) for i in np.ravel(pose.rotation, order='F').tolist()],
                    "center": [str(i) for i in pose.center]
                },
                "locked": "0"
            }
        }

        return d

    # Construct AliceVision struct
    data = {
        "version": ["1", "0", "0"],
        "views": [av_view(view) for view in sfm.views],
        "intrinsics": [av_intrinsic(intr) for intr in sfm.intrinsics],
        "poses": [av_pose(pose) for pose in sfm.poses]
    }

    # Write to a JSON file
    output_file = Path(path)
    with output_file.open(mode='w') as f:
        json.dump(data, f, indent=4)
