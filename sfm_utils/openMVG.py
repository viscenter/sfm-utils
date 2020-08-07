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

import json
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np

from sfm_utils.sfm import Intrinsic, IntrinsicType, Pose, SfMScene, View

__OPENMVG_CAMDB_DEFAULT_PATH = '/usr/local/share/openMVG/sensor_width_camera_database.txt'

__OPENMVG_ROT_MAT = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

__OPENMVG_INTRINSIC_NAME_MAP = {
    IntrinsicType.PINHOLE: 'pinhole',
    IntrinsicType.RADIAL_K3: 'pinhole_radial_k3',
    IntrinsicType.BROWN_T2: 'pinhole_brown_t2'
}

__OPENMVG_DIST_NAME_MAP = {
    IntrinsicType.RADIAL_K3: 'disto_k3',
    IntrinsicType.BROWN_T2: 'disto_t2'
}


def load_cam_db(db_path: Union[str, bytes, PathLike, None] = None) -> dict:
    """
    Load the OpenMVG camera database file
    """
    # Make into a Path
    if db_path is None:
        file_path = Path(__OPENMVG_CAMDB_DEFAULT_PATH)
    else:
        file_path = Path(db_path)

    # Load the cam db
    d = {}
    with file_path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.count(';') != 1:
                print(f'ERROR: Cannot parse CamDB entry: {line}')
                continue
            key, value = line.split(';')
            d[key] = float(value)
    return d


def export_openmvg(path: Union[str, bytes, PathLike], sfm: SfMScene):
    """
    Export SfMScene to an OpenMVG JSON file
    """
    # Emulate the Cereal pointer counter
    __ptr_cnt = 2147483649

    def open_mvg_view(view: View) -> dict:
        """
        OpenMVG View struct
        """
        nonlocal __ptr_cnt
        d = {
            "key": view.id,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": __ptr_cnt,
                    "data": {
                        "local_path": '',
                        "filename": str(view.path.name),
                        "width": view.width,
                        "height": view.height,
                        "id_view": view.id,
                        "id_intrinsic": view.intrinsic.id,
                        "id_pose": view.pose.id,
                    }
                }
            }
        }
        __ptr_cnt += 1
        return d

    def open_mvg_intrinsic(intrinsic: Intrinsic) -> dict:
        """
        OpenMVG Intrinsic struct
        """
        nonlocal __ptr_cnt
        d = {
            'key': intrinsic.id,
            'value': {
                'polymorphic_id': 2147483649,
                "polymorphic_name": __OPENMVG_INTRINSIC_NAME_MAP[intrinsic.type],
                "ptr_wrapper": {
                    "id": __ptr_cnt,
                    "data": {
                        "width": intrinsic.width,
                        "height": intrinsic.height,
                        "focal_length": intrinsic.focal_length_as_pixels,
                        "principal_point": [
                            intrinsic.ppx,
                            intrinsic.ppy
                        ]
                    }
                }
            }
        }
        __ptr_cnt += 1

        if intrinsic.dist_params is not None:
            dist_name = __OPENMVG_DIST_NAME_MAP[intrinsic.type]
            d['value']['ptr_wrapper']['data'][dist_name] = intrinsic.dist_params

        return d

    def open_mvg_extrinsic(extrinsic: Pose) -> dict:
        """
        OpenMVG Extrinsic struct
        """
        d = {
            "key": extrinsic.id,
            "value": {
                "rotation": (__OPENMVG_ROT_MAT @ extrinsic.rotation).tolist(),
                "center": extrinsic.center
            }
        }
        return d

    # Construct OpenMVG struct
    data = {
        'sfm_data_version': '0.3',
        'root_path': str(sfm.root_dir),
        'views': [open_mvg_view(view) for view in sfm.views],
        'intrinsics': [open_mvg_intrinsic(intr) for intr in sfm.intrinsics],
        'extrinsics': [open_mvg_extrinsic(extr) for extr in sfm.poses],
        'structure': [],
        'control_points': []
    }

    # Write to disk
    output_file = Path(path)
    with output_file.open(mode='w') as f:
        json.dump(data, f, indent=4)
