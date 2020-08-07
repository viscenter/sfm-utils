"""
PySfMUtils
Copyright (C) 2020  EduceLab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

from sfm_utils.sfm import Intrinsic, IntrinsicType, Pose, Scene, View

__AV_INTRINSIC_NAME_MAP = {
    IntrinsicType.PINHOLE: 'pinhole',
    IntrinsicType.RADIAL_K3: 'radial3',
    IntrinsicType.BROWN_T2: 'brownt2'
}


def scene_to_alicevision(scene: Scene):
    """
    Convert Scene to an AliceVision-formatted dict. This dict can be written to a project file with the json package.

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
        "views": [av_view(view) for view in scene.views],
        "intrinsics": [av_intrinsic(intr) for intr in scene.intrinsics],
        "poses": [av_pose(pose) for pose in scene.poses]
    }

    return data
