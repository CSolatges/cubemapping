import unittest

from codes.eq_rec_to_cube import from_equi_xy_to_cube_xyz, get_closest_2_pow_i_multiple_of_1024_from_height


class TestEqRec2Cube(unittest.TestCase):
    def test_exemple(self) -> None:
        self.assertTrue(True)

    def test_out_img_to_xyz(self) -> None:
        self.assertEqual(from_equi_xy_to_cube_xyz(
            0, 0, 'u', 128), (-1.0, -5.0, 1.0))
        self.assertEqual(from_equi_xy_to_cube_xyz(
            128, 64, 'r', 128), (5.0, 1.0, 2.0))
        self.assertEqual(from_equi_xy_to_cube_xyz(
            64, 64, 'f', 128), (1.0, -4.0, 2.0))
        self.assertEqual(from_equi_xy_to_cube_xyz(
            192, 64, 'b', 128), (-1.0, -2.0, 2.0))
        self.assertEqual(from_equi_xy_to_cube_xyz(
            0, 64, 'l', 128), (-3.0, -1.0, 2.0))
        self.assertEqual(from_equi_xy_to_cube_xyz(
            128, 128, 'd', 128), (3.0, -3.0, -1.0))

    def test_get_closest_2_pow_i_multiple_of_1024_from_height(self) -> None:
        self.assertEqual(
            get_closest_2_pow_i_multiple_of_1024_from_height(4096), (3, 8192))
        self.assertEqual(
            get_closest_2_pow_i_multiple_of_1024_from_height(4095), (2, 4096))
        self.assertEqual(
            get_closest_2_pow_i_multiple_of_1024_from_height(4097), (3, 8192))


if __name__ == '__main__':
    unittest.main()
