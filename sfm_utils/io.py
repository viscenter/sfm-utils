import json
from enum import auto, Enum
from os import PathLike
from pathlib import Path
from typing import Union

from sfm_utils.alicevision import scene_to_alicevision
from sfm_utils.openmvg import scene_to_openmvg
from sfm_utils.sfm import Scene


class Format(Enum):
    """
    SfM file formats
    """
    OPEN_MVG = auto()
    ALICE_VISION = auto()


def export_scene(path: Union[str, bytes, PathLike], scene: Scene, fmt: Format = Format.OPEN_MVG,
                 convert_rotations: bool = True):
    """
    Export Scene to a project file
    """
    if fmt == Format.OPEN_MVG:
        data = scene_to_openmvg(scene, convert_rotations=convert_rotations)
    elif fmt == Format.ALICE_VISION:
        data = scene_to_alicevision(scene)
    else:
        raise ValueError('Unknown scene format')

    # Write to disk
    output_file = Path(path)
    with output_file.open(mode='w') as f:
        json.dump(data, f, indent=4)
