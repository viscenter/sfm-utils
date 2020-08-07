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

from abc import ABC
from enum import auto, Enum
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


class IntrinsicType(Enum):
    PINHOLE = auto()
    RADIAL_K3 = auto()
    BROWN_T2 = auto()


class SceneElement(ABC):
    """
    Base class for SfM scene elements
    """
    _id = None

    @property
    def id(self):
        """
        Identifier within the SfM scene
        """
        return self._id

    @id.setter
    def id(self, i: int):
        self._id = i


class Intrinsic(SceneElement):
    """
    Basic pinhole camera intrinsic
    """
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
    def type(self) -> IntrinsicType:
        """
        Intrinsic type
        """
        return IntrinsicType.PINHOLE

    @property
    def width(self) -> int:
        """
        Image width, in pixels
        """
        return self._image_width

    @width.setter
    def width(self, w: int):
        self._image_width = w

    @property
    def height(self) -> int:
        """
        Image height, in pixels
        """
        return self._image_height

    @height.setter
    def height(self, h: int):
        self._image_height = h

    @property
    def ppx(self) -> float:
        """
        Principal point X position, in pixels
        """
        if self._ppx is None:
            return self.width / 2.0
        else:
            return self._ppx

    @ppx.setter
    def ppx(self, x: float):
        self._ppx = x

    @property
    def ppy(self) -> float:
        """
        Principal point Y position, in pixels
        """
        if self._ppy is None:
            return self.height / 2.0
        else:
            return self._ppy

    @ppy.setter
    def ppy(self, y: float):
        self._ppy = y

    @property
    def principal_point(self) -> Tuple[float, float]:
        """
        Principal point as a tuple
        """
        return self.ppx, self.ppy

    @property
    def focal_length(self) -> float:
        """
        Effective lens focal length, in mm
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, f: float):
        self._focal_length = f

    @property
    def focal_length_as_pixels(self) -> float:
        """
        Effective lens focal length, in pixels
        """
        if self._focal_length_as_pixels is None:
            return max(self.width, self.height) * self.focal_length / self.sensor_width
        else:
            return self._focal_length_as_pixels

    @focal_length_as_pixels.setter
    def focal_length_as_pixels(self, f: float):
        self._focal_length_as_pixels = f

    @property
    def sensor_width(self) -> float:
        """
        Largest image sensor dimension (typically the sensor width), in mm.
        :return:
        """
        return self._sensor_width

    @sensor_width.setter
    def sensor_width(self, w: float):
        self._sensor_width = w

    @property
    def dist_params(self) -> List[float]:
        """
        Lens distortion parameters
        """
        return self._dist_params


class IntrinsicRadialK3(Intrinsic):
    """
    Pinhole intrinsic with three radial distortion coefficients
    """

    def __init__(self):
        super().__init__()
        self._dist_params = [0.0, 0.0, 0.0]

    def __eq__(self, other):
        if super().__eq__(other):
            return self.dist_params == other.dist_params
        else:
            return False

    @property
    def type(self) -> IntrinsicType:
        """
        Intrinsic type
        """
        return IntrinsicType.RADIAL_K3

    @Intrinsic.dist_params.setter
    def dist_params(self, p: List[float]):
        self._dist_params = p[0:3]


class IntrinsicBrownT2(Intrinsic):
    """
    Pinhole intrinsic with three radial distortion and two tangential coefficients
    """

    def __init__(self):
        super().__init__()
        self._dist_params = [0.0, 0.0, 0.0, 0.0, 0.0]

    def __eq__(self, other):
        if super().__eq__(other):
            return self.dist_params == other.dist_params
        else:
            return False

    @property
    def type(self) -> IntrinsicType:
        """
        Intrinsic type
        """
        return IntrinsicType.BROWN_T2

    @Intrinsic.dist_params.setter
    def dist_params(self, p: List[float]):
        self._dist_params = p[0:5]


class Pose(SceneElement):
    """
    Camera pose as center position and rotation matrix. This package assumes that rotations use a right-handed system.
    """
    _center = None
    _rotation = None

    @property
    def center(self) -> List[float]:
        """
        Camera position in world coordinates
        """
        if self._center is None:
            return [0., 0., 0.]
        else:
            return self._center

    @center.setter
    def center(self, c: List[float]):
        self._center = c[0:3]

    @property
    def rotation(self) -> np.ndarray:
        """
        Camera rotation matrix
        """
        if self._rotation is None:
            return np.eye(3)
        else:
            return self._rotation

    @rotation.setter
    def rotation(self, r: np.ndarray):
        self._rotation = r


class View(SceneElement):
    """
    Observation of scene (i.e. image file) and related parameters
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
        """
        Path to the image file
        """
        return self._path

    @path.setter
    def path(self, path: Union[str, bytes, PathLike]):
        self._path = Path(path)

    @property
    def intrinsic(self) -> Intrinsic:
        """
        View camera intrinsics
        """
        return self._intrinsic

    @intrinsic.setter
    def intrinsic(self, i: Intrinsic):
        self._intrinsic = i

    @property
    def pose(self) -> Pose:
        """
        View camera pose
        """
        return self._pose

    @pose.setter
    def pose(self, p: Pose):
        self._pose = p

    @property
    def width(self) -> int:
        """
        Image width, in pixels
        """
        return self._image_width

    @width.setter
    def width(self, w: int):
        self._image_width = w

    @property
    def height(self) -> int:
        """
        Image height, in pixels
        """
        return self._image_height

    @height.setter
    def height(self, h: int):
        self._image_height = h

    @property
    def camera_make(self) -> str:
        """
        View camera make as string
        """
        return self._make

    @camera_make.setter
    def camera_make(self, m: str):
        self._make = m

    @property
    def camera_model(self) -> str:
        """
        View camera model as string
        """
        return self._model

    @camera_model.setter
    def camera_model(self, m: str):
        self._model = m

    @property
    def camera_make_model(self) -> str:
        """
        View camera make and model as single string
        """
        return f'{self.camera_make} {self.camera_model}'


class Scene:
    """
    SfM scene
    """
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
        """
        Root directory for scene files
        """
        return self._root

    @root_dir.setter
    def root_dir(self, p: Union[str, bytes, PathLike]):
        self._root = Path(p)

    @property
    def views(self) -> List[View]:
        """
        List of all views
        """
        return self._views

    @views.setter
    def views(self, views: List[View]):
        self._views = views

    def add_view(self, view: View) -> View:
        """
        Add a View to the scene and return the added View.
        """
        view.id = len(self._views)
        self._views.append(view)
        return view

    @property
    def intrinsics(self) -> List[Intrinsic]:
        """
        List of all intrinsics
        """
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics: List[Intrinsic]):
        self._intrinsics = intrinsics

    def add_intrinsic(self, intrinsic: Intrinsic, group_models: bool = True) -> Intrinsic:
        """
        Add an Intrinsic to the scene and return the added Intrinsic. The returned Intrinsic should be assigned to the
        associated View object. If `group_models` is True and the passed Intrinsic is identical to one already in the
        scene, the passed Intrinsic is ignored and the existing one is returned.
        """
        # See if there's an identical intrinsic already
        if group_models:
            for existing in self._intrinsics:
                if intrinsic == existing:
                    return existing

        # If not, make a new one
        intrinsic.id = len(self._intrinsics)
        self._intrinsics.append(intrinsic)
        return intrinsic

    @property
    def poses(self) -> List[Pose]:
        """
        List of all poses
        """
        return self._poses

    @poses.setter
    def poses(self, p: List[Pose]):
        self._poses = p

    def add_pose(self, pose: Pose) -> Pose:
        """
        Add a Pose to the scene and return the added Pose
        """
        pose.id = len(self._poses)
        self._poses.append(pose)
        return pose
