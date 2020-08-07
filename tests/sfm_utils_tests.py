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

import os
import unittest
from pathlib import Path

import sfm_utils as sfm


class TestIntrinsics(unittest.TestCase):
    def test_defaults(self):
        intrinsic = sfm.Intrinsic()
        self.assertEqual(intrinsic.type, sfm.IntrinsicType.PINHOLE)
        self.assertEqual(intrinsic.width, None)
        self.assertEqual(intrinsic.height, None)
        self.assertEqual(intrinsic.focal_length, None)
        self.assertEqual(intrinsic.sensor_width, None)
        self.assertEqual(intrinsic.dist_params, None)
        with self.assertRaises(TypeError):
            _ = intrinsic.ppx
            _ = intrinsic.ppy
            _ = intrinsic.principal_point
            _ = intrinsic.focal_length_as_pixels

    def test_properties(self):
        intrinsic = sfm.Intrinsic()

        # Assign to properties without auto values
        intrinsic.width = 100
        intrinsic.height = 100
        intrinsic.focal_length = 10
        intrinsic.sensor_width = 10

        # Test property assignment
        self.assertEqual(intrinsic.width, 100)
        self.assertEqual(intrinsic.height, 100)
        self.assertEqual(intrinsic.focal_length, 10)
        self.assertEqual(intrinsic.sensor_width, 10)

        # Test auto-property values
        self.assertEqual(intrinsic.ppx, 50)
        self.assertEqual(intrinsic.ppy, 50)
        self.assertEqual(intrinsic.principal_point, (50, 50))
        self.assertEqual(intrinsic.focal_length_as_pixels, 100)

        # Override and test auto-properties
        intrinsic.ppx = 49
        intrinsic.ppy = 49
        intrinsic.focal_length_as_pixels = 50
        self.assertEqual(intrinsic.ppx, 49)
        self.assertEqual(intrinsic.ppy, 49)
        self.assertEqual(intrinsic.principal_point, (49, 49))
        self.assertEqual(intrinsic.focal_length_as_pixels, 50)

    def test_radial_k3(self):
        # Test defaults
        intrinsic = sfm.IntrinsicRadialK3()
        self.assertEqual(intrinsic.type, sfm.IntrinsicType.RADIAL_K3)
        self.assertEqual(intrinsic.dist_params, [0.0, 0.0, 0.0])

        # Override dist params
        intrinsic.dist_params = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(intrinsic.dist_params, [0.0, 1.0, 2.0])

    def test_brown_t2(self):
        # Test defaults
        intrinsic = sfm.IntrinsicBrownT2()
        self.assertEqual(intrinsic.type, sfm.IntrinsicType.BROWN_T2)
        self.assertEqual(intrinsic.dist_params, [0.0, 0.0, 0.0, 0.0, 0.0])

        # Override dist params
        intrinsic.dist_params = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(intrinsic.dist_params, [0.0, 1.0, 2.0, 3.0, 4.0])


class TestIO(unittest.TestCase):
    def setUp(self) -> None:
        # Create scene
        self.scene = sfm.Scene()
        self.scene.root_dir = '.'

        # Construct a view
        view = sfm.View()
        view.path = "view.jpg"
        view.width = 100
        view.height = 100
        view.camera_make = 'PySfMUtils'
        view.camera_model = 'Test Camera'
        self.scene.add_view(view)

        # Add the view intrinsic
        intrinsic = sfm.Intrinsic()
        intrinsic.width = view.width
        intrinsic.height = view.height
        intrinsic.focal_length = 10
        intrinsic.sensor_width = 10
        view.intrinsic = self.scene.add_intrinsic(intrinsic)

        # Add the view pose
        pose = sfm.Pose()
        view.pose = self.scene.add_pose(pose)

    def test_export_scene(self):
        import json
        # Directory of expected files
        expected_dir = Path(os.path.realpath(__file__)).parent / 'expected'

        # Compare the OpenMVG results
        with Path(expected_dir / 'openmvg_sfm.json').open('r') as f:
            openmvg_expected = f.read()
        openmvg_result = json.dumps(sfm.scene_to_openmvg(self.scene), indent=4)
        self.assertEqual(openmvg_expected, openmvg_result)

        # Compare the AliceVision results
        with Path(expected_dir / 'alicevision_sfm.json').open('r') as f:
            av_expected = f.read()
        av_result = json.dumps(sfm.scene_to_alicevision(self.scene), indent=4)
        self.assertEqual(av_expected, av_result)


if __name__ == '__main__':
    unittest.main()
