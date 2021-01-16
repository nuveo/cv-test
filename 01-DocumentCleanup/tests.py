import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import unittest

import cleaner


class TestCleanerMethods(unittest.TestCase):

    def test_shape(self):
        test_image = cv2.imread("noisy_data/2.png")
        result = cleaner.clean_document(test_image)
        self.assertEqual(result.shape, test_image.shape[:2])

    def test_is_binary(self):
        test_image = cv2.imread("noisy_data/2.png")
        result = cleaner.clean_document(test_image)
        valid_values = (result == 0) | (result == 255)
        self.assertTrue(valid_values.all())

    def test_correct_skew_shape(self):
        test_image = cv2.imread("noisy_data/2.png")
        binary_image = cleaner.binarize_image(test_image)
        result = cleaner.correct_skew(binary_image)
        self.assertEqual(result.shape, test_image.shape[:2])

    def test_correct_skew_is_binary(self):
        test_image = cv2.imread("noisy_data/2.png")
        binary_image = cleaner.binarize_image(test_image)
        result = cleaner.correct_skew(binary_image)
        valid_values = (result == 0) | (result == 255)
        self.assertTrue(valid_values.all())

    def test_skew_score(self):
        test_image_aligned = self.create_striped_image()

        rotated_minus_10 = inter.rotate(test_image_aligned, -10, reshape=False, order=0)
        rotated_10 = inter.rotate(test_image_aligned, -10, reshape=False, order=0)

        score_aligned = cleaner.calculate_skew_score(test_image_aligned)
        score_minus_10 = cleaner.calculate_skew_score(rotated_minus_10)
        score_10 = cleaner.calculate_skew_score(rotated_10)

        self.assertGreater(score_aligned, score_minus_10)
        self.assertGreater(score_aligned, score_10)

    def create_striped_image(self):
        test_image_aligned = np.zeros((100, 100), dtype=np.uint8)
        test_image_aligned[0:10, :] = 255
        test_image_aligned[20:30, :] = 255
        test_image_aligned[40:50, :] = 255
        test_image_aligned[60:70, :] = 255
        test_image_aligned[80:90, :] = 255
        return test_image_aligned

    def test_invert_binary_image(self):
        test_image = self.create_striped_image()
        inverted = cleaner.invert_binary_image(test_image)
        originally_black = test_image == 0
        originally_white = test_image == 255
        inverted_black = inverted == 0
        inverted_white = inverted == 255
        
        self.assertTrue((originally_black == inverted_white).all())
        self.assertTrue((originally_white == inverted_black).all())

    def test_remove_noise_based_on_histogram(self):
        test_image = self.create_striped_image()
        test_image[5, 5] = 0
        test_image[5, 0] = 0
        test_image[5, 50] = 0
        test_image[50, 50] = 0
        test_image[65, 2] = 0

        result = cleaner.remove_noise_based_on_histogram(test_image)

        self.assertEqual(result[5, 5], 255)
        self.assertEqual(result[5, 0], 255)
        self.assertEqual(result[5, 50], 255)
        self.assertEqual(result[40, 40], 255)
        self.assertEqual(result[65, 2], 255)
