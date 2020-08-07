import numpy as np
from os import PathLike
from typing import Union
from pathlib import Path

__OPENMVG_CAMDB_DEFAULT_PATH = '/usr/local/share/openMVG/sensor_width_camera_database.txt'
__OPENMVG_ROT_MAT = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def load_cam_db(db_path: Union[str, bytes, PathLike, None] = None) -> dict:
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

