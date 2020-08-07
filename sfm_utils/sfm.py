import argparse
import json
import re
from enum import auto, Enum
from pathlib import Path
from typing import List
from os import PathLike

import exiftool
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class SfMFormat(Enum):
    """
    SfM file formats
    """
    OPEN_MVG = auto()
    ALICE_VISION = auto()


class SfMSceneElement:
    """
    Base class for SfM scene elements
    """
    _id = None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i: int):
        self._id = i

class Intrinsic(SfMSceneElement):
    _image_width = None
    _image_height = None
    _ppx = None
    _ppy = None
    _focal_length = None
    _focal_length_as_pixels = None
    _sensor_width = None
    _dist_params = None

    def __eq__(self, other):
        if not isinstance(other, Intrinsic) or self.type != other.type:
            return False

        eq = self.width == other.width
        eq = eq and self.height == other.height
        eq = eq and self.ppx == other.ppx
        eq = eq and self.ppy == other.ppy
        eq = eq and self.focal_length_as_pixels == other.focal_length_as_pixels
        return eq

    @property
    def type(self) -> str:
        return 'pinhole'

    @property
    def width(self) -> int:
        return self._image_width

    @width.setter
    def width(self, w: int):
        self._image_width = w

    @property
    def height(self) -> int:
        return self._image_height

    @height.setter
    def height(self, h: int):
        self._image_height = h

    @property
    def ppx(self) -> float:
        if self._ppx is None:
            return self.width / 2.0
        else:
            return self._ppx

    @ppx.setter
    def ppx(self, x: float):
        self._ppx = x

    @property
    def ppy(self) -> float:
        if self._ppy is None:
            return self.height / 2.0
        else:
            return self._ppy

    @ppy.setter
    def ppy(self, y: float):
        self._ppy = y

    @property
    def focal_length(self) -> float:
        return self._focal_length

    @focal_length.setter
    def focal_length(self, f: float):
        self._focal_length = f

    @property
    def focal_length_as_pixels(self) -> float:
        if self._focal_length_as_pixels is None:
            return max(self.width, self.height) * self.focal_length / self.sensor_width
        else:
            return self._focal_length_as_pixels

    @focal_length_as_pixels.setter
    def focal_length_as_pixels(self, f: float):
        self._focal_length_as_pixels = f

    @property
    def sensor_width(self) -> float:
        return self._sensor_width

    @sensor_width.setter
    def sensor_width(self, w: float):
        self._sensor_width = w

    @property
    def dist_params(self) -> List[float]:
        return self._dist_params


class IntrinsicRadialK3(Intrinsic):
    def __init__(self):
        super().__init__()
        self._dist_params = [0.0, 0.0, 0.0]

    def __eq__(self, other):
        if super().__eq__(other):
            return self.dist_params == other.dist_params
        else:
            return False

    @property
    def type(self) -> str:
        return 'radial3'

    @Intrinsic.dist_params.setter
    def dist_params(self, p: List[float]):
        self._dist_params = p[0:3]


class IntrinsicBrownT2(Intrinsic):
    def __init__(self):
        super().__init__()
        self._dist_params = [0.0, 0.0, 0.0, 0.0, 0.0]

    def __eq__(self, other):
        if super().__eq__(other):
            return self.dist_params == other.dist_params
        else:
            return False

    @property
    def type(self) -> str:
        return 'brownt2'

    @Intrinsic.dist_params.setter
    def dist_params(self, p: List[float]):
        self._dist_params = p[0:5]


class Pose(SfMSceneElement):
    _center = None
    _rotation = None

    @property
    def center(self) -> List[float]:
        if self._center is None:
            return [0., 0., 0.]
        else:
            return self._center

    @center.setter
    def center(self, c: List[float]):
        self._center = c[0:3]

    @property
    def rotation(self) -> np.ndarray:
        if self._rotation is None:
            return np.eye(3, 3)
        else:
            return self._rotation

    @rotation.setter
    def rotation(self, r: np.ndarray):
        self._rotation = r


class View(SfMSceneElement):
    """

    """
    _path = None
    _intrinsic = None
    _pose = None
    _image_width = None
    _image_height = None
    _make = None
    _model = None

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = Path(path)

    @property
    def intrinsic(self) -> Intrinsic:
        return self._intrinsic

    @intrinsic.setter
    def intrinsic(self, i: Intrinsic):
        self._intrinsic = i

    @property
    def pose_id(self) -> int:
        return self._pose_id

    @pose_id.setter
    def pose_id(self, i: int):
        self._pose_id = i

    @property
    def width(self) -> int:
        return self._image_width

    @width.setter
    def width(self, w: int):
        self._image_width = w

    @property
    def height(self) -> int:
        return self._image_height

    @height.setter
    def height(self, h: int):
        self._image_height = h

    @property
    def make(self) -> str:
        return self._make

    @make.setter
    def make(self, m: str):
        self._make = m

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, m: str):
        self._model = m


class SfMScene:
    _root = None
    _views = None
    _intrinsics = None
    _poses = None

    def __init__(self):
        self._views = []
        self._intrinsics = []
        self._poses = []

    @property
    def root_dir(self) -> Path:
        return self._root

    @root_dir.setter
    def root_dir(self, p: Path):
        self._root = Path(p)

    @property
    def views(self) -> List[View]:
        return self._views

    @views.setter
    def views(self, views: List[View]):
        self._views = views

    def add_view(self, view: View) -> int:
        view.id = len(self._views)
        self._views.append(view)
        return view.id

    @property
    def intrinsics(self) -> List[Intrinsic]:
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics: List[Intrinsic]):
        self._intrinsics = intrinsics

    def add_intrinsic(self, intrinsic: Intrinsic) -> int:
        # See if there's an identical intrinsic already
        for existing in self._intrinsics:
            if intrinsic == existing:
                return existing.id

        # If not, make a new one
        intrinsic.id = len(self._intrinsics)
        self._intrinsics.append(intrinsic)
        return intrinsic.id

    @property
    def poses(self) -> List[Pose]:
        return self._poses

    @poses.setter
    def poses(self, p: List[Pose]):
        self._poses = p

    def add_pose(self, pose: Pose) -> int:
        pose.id = len(self._poses)
        self._poses.append(pose)
        return pose.id

    def save(self, path: str, fmt: SfMFormat = SfMFormat.OPEN_MVG):
        sfm_export(path, self, fmt)


def import_pgs_scan(scan_dir: Path, cam_db: dict, cam_calib: dict = None) -> SfMScene:
    # Load scan metadata
    meta_path = scan_dir / 'metadata.json'
    with meta_path.open() as f:
        scan_meta = json.loads(f.read())

    # Insert calibration data into camera metadata
    if cam_calib is not None:
        for cam in scan_meta['scanner']['cameras']:
            if 's/n' in cam.keys():
                serial_no = cam['s/n']
                if serial_no in cam_calib['calibs'].keys():
                    cam['k3'] = cam_calib['calibs'][serial_no]['k3']
                    # cam['t2'] = cam_calib['calibs'][serial_no]['t2']

    # Get list of images
    prefix = scan_meta['scan']['file_prefix']
    ext = scan_meta['scan']['format'].lower()
    images = list(scan_dir.glob(f'{prefix}*.{ext}'))
    images.sort()

    # Get image metadata
    files = [str(i) for i in images]
    if len(files) == 0:
        print('Error: Provided scan metadata specifies file pattern, but no files match.')
        raise RuntimeError()

    with exiftool.ExifTool() as et:
        img_metadata = et.get_metadata_batch(files)

    # Setup sfm
    sfm = SfMScene()
    sfm.root_dir = scan_dir

    # Fill out sfm with data
    for img in images:
        # Lookup this images tags
        tags = next((i for i in img_metadata if i['File:FileName'] == img.name), None)
        if tags is None:
            print(f'Error: No tags loaded for image: {str(img)}')
            continue

        # Setup view
        view = View()
        view.path = img
        view.width = tags['File:ImageWidth']
        view.height = tags['File:ImageHeight']
        view.make = tags['EXIF:Make']
        view.model = tags['EXIF:Model']

        # Get the camera idx and the position idx
        cam_idx = None
        pos_idx = None
        if re.fullmatch(rf'{re.escape(prefix)}\d*_\d*\.{re.escape(ext)}', img.name):
            cam_idx, pos_idx = img.name.replace(prefix, '').replace(f'.{ext}', '').split('_')
            cam_idx = int(cam_idx)
            pos_idx = int(pos_idx)

        # Setup intrinsic
        intrinsic = IntrinsicRadialK3()
        if cam_idx is not None:
            cam = scan_meta['scanner']['cameras'][cam_idx]
            if 'k3' in cam.keys() and 't2' in cam.keys():
                intrinsic = IntrinsicBrownT2()
                intrinsic.dist_params = cam['k3'] + cam['t2']
            elif 'k3' in cam.keys():
                intrinsic.dist_params = cam['k3']
        intrinsic.width = view.width
        intrinsic.height = view.height
        intrinsic.focal_length = tags['EXIF:FocalLength']
        if f'{view.make} {view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.make} {view.model}']
        elif f'{view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.model}']
        else:
            print(f'Camera not in database: {view.make} {view.model}. Ignoring file: {img.name}')
            continue

        # Init extrinsics
        pose = Pose()
        if cam_idx is not None:
            # Get Camera
            cam = scan_meta['scanner']['cameras'][cam_idx]

            # Assign position
            if cam['is_absolute_pos'] is True or pos_idx is None:
                if pos_idx is None:
                    print(f'Couldn\'t parse position index. Interpreting file\'s pose as absolute: {img.name}')
                pose.center = cam['position']
            else:
                center = np.array(scan_meta['scan']['capture_positions'][pos_idx])
                offset = cam['position']
                position = np.add(center, offset)
                pose.center = position.round(15).tolist()

            # Calculate the rotation matrix
            # Our rotation matrix is right handed and row-major
            # We compose rotations as ZYX, so reverse the angle list
            euler_angles = cam['rotation'][::-1]
            rotation = Rot.from_euler('zyx', euler_angles, degrees=True)
            pose.rotation = rotation.as_matrix().round(15)

        # Only add everything to the SfM at the end
        sfm.add_view(view)
        view.intrinsic_id = sfm.add_intrinsic(intrinsic)
        view.pose_id = sfm.add_pose(pose)

    # Return the filled out sfm
    return sfm


def sfm_export(path: str, sfm: SfMScene, fmt: SfMFormat = SfMFormat.OPEN_MVG):
    if fmt == SfMFormat.OPEN_MVG:
        sfm_export_openmvg(path, sfm)
    elif fmt == SfMFormat.ALICE_VISION:
        sfm_export_alicevision(path, sfm)


def sfm_export_openmvg(path: str, sfm: SfMScene):
    __ptr_cnt = 2147483649

    def open_mvg_view(view: View) -> dict:
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
                        "id_intrinsic": view.intrinsic_id,
                        "id_pose": view.pose_id,
                    }
                }
            }
        }
        __ptr_cnt += 1
        return d

    def open_mvg_intrinsic(intrinsic: Intrinsic) -> dict:
        name = None
        dist_key = None
        dist = None
        if type(intrinsic) is Intrinsic:
            name = 'pinhole'
        elif type(intrinsic) is IntrinsicRadialK3:
            name = 'pinhole_radial_k3'
            dist_key = 'disto_k3'
            dist = intrinsic.dist_params
        elif type(intrinsic) is IntrinsicBrownT2:
            name = 'pinhole_brown_t2'
            dist_key = 'disto_t2'
            dist = intrinsic.dist_params

        nonlocal __ptr_cnt
        d = {
            'key': intrinsic.id,
            'value': {
                'polymorphic_id': 2147483649,
                "polymorphic_name": name,
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

        if dist_key is not None and dist is not None:
            d['value']['ptr_wrapper']['data'][dist_key] = dist

        return d

    def open_mvg_extrinsic(extrinsic: Pose) -> dict:
        d = {
            "key": extrinsic.id,
            "value": {
                "rotation": (__PGS_TO_OPENMVG_ROT_MAT @ extrinsic.rotation).tolist(),
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

    output_file = Path(path)
    with output_file.open(mode='w') as f:
        json.dump(data, f, indent=4)


def sfm_export_alicevision(path: str, sfm: SfMScene):
    def av_view(view: View):
        d = {
            "viewID": str(view.id),
            "poseID": str(view.pose_id),
            "intrinsicID": str(view.intrinsic_id),
            "path": str(view.path),
            "width": str(view.width),
            "height": str(view.height)
        }
        return d

    def av_intrinsic(intrinsic: Intrinsic):
        dist = None
        if type(intrinsic) is IntrinsicRadialK3:
            dist = [str(i) for i in intrinsic.dist_params]

        d = {
            "intrinsicID": str(intrinsic.id),
            "width": str(intrinsic.width),
            "height": str(intrinsic.height),
            "serialNumber": str(intrinsic.id),
            "type": intrinsic.type,
            "initializationMode": "estimated",
            "pxInitialFocalLength": str(intrinsic.focal_length_as_pixels),
            "pxFocalLength": str(intrinsic.focal_length_as_pixels),
            "principalPoint": [
                str(intrinsic.ppx),
                str(intrinsic.ppy)
            ],
            "locked": "0"
        }

        if dist is not None:
            d['distortionParams'] = dist

        return d

    def av_pose(pose: Pose):
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

    data = {
        "version": ["1", "0", "0"],
        "views": [av_view(view) for view in sfm.views],
        "intrinsics": [av_intrinsic(intr) for intr in sfm.intrinsics],
        "poses": [av_pose(pose) for pose in sfm.poses]
    }

    output_file = Path(path)
    with output_file.open(mode='w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pgs-dir', '-p', required=True, help='PGS Scan directory')
    parser.add_argument('--cam-db', '-d', default=__CAMERA_FILE_PARAMS, help='Camera database path')
    parser.add_argument('--cam-calib', '-c', help="Camera calibrations file")
    parser.add_argument('--output-sfm', '-o', default='sfm_data.json', help='Output SFM file')
    args = parser.parse_args()

    # Load the camera db
    cam_db_path = Path(args.cam_db)
    cam_db = load_cam_db(cam_db_path)

    # Load the camera calibrations (if present)
    calib = None
    if args.cam_calib:
        print('Loading camera calibrations...')
        calib_path = Path(args.cam_calib)
        calib = load_cam_calib(calib_path)

    # Load the pgs file
    print('Loading PGS Scan...')
    pgs_dir_path = Path(args.pgs_dir)
    sfm = import_pgs_scan(pgs_dir_path, cam_db, calib)

    # Write the SFM
    print('Exporting SfM scene...')
    sfm.save(args.output_sfm)

    print('Done.')


if __name__ == '__main__':
    main()
