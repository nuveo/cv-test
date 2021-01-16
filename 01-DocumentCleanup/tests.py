import cv2
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
