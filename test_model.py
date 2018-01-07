import unittest
import os
from model import *
import cv2


class SDCSimulationTrain(unittest.TestCase):

    def test_tensorflow_tags(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.assertEqual(os.environ['TF_CPP_MIN_LOG_LEVEL'], '2')

    def test_load_configuration(self):
        conf_path = 'test/test_configuration.json'
        tester_configuration = {"input_path": "test/test_data",
                                "output_path": "test/test_logs/test_model",
                                "loss_function": "mse",
                                "epochs": 3,
                                "use_tensorboard": "True",
                                "units": 1,
                                "tensorboard_log_dir": "test/test_logs",
                                "old_image_root": "test/test_data",
                                "new_image_root": "test/test_data",
                                "checkpoint_path": "test/test_logs",
                                "gpus": 1,
                                "batch_size": 2,
                                "learning_rate": 0.00005,
                                "test_size": 0.2,
                                "dropout_percentage": 0.3,
                                "side_adjustment": 0.35
                                }
        loaded_configuration = load_config(conf_path)
        self.assertDictEqual(loaded_configuration, tester_configuration)

    def test_get_file_list(self):
        directory_path = 'test/test_data'
        test_file_list = ['test/test_data/inside_in_grass_fast/driving_log.csv',
                      'test/test_data/inside_just_at_curb_good/driving_log.csv']
        file_list = get_file_list(directory_path)
        self.assertEqual(file_list, test_file_list)

    def test_get_log_lines(self):
        test_file_list = ['test/test_data/inside_in_grass_fast/driving_log.csv',
                      'test/test_data/inside_just_at_curb_good/driving_log.csv']
        test_lines = []
        [test_lines.append([path, get_log_lines(path)]) for path in test_file_list]
        reference_value = '30.19097'
        self.assertEqual(test_lines[1][1][1][6], reference_value)


    def test_augment_brightness_camera_images(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        brightened_image = augment_brightness_camera_images(image)
        # Assert that the same shape was returned. Can do more intensive checks later.
        self.assertEqual(image.shape[0], brightened_image.shape[0])
        self.assertEqual(image.shape[1], brightened_image.shape[1])
        self.assertEqual(image.shape[2], brightened_image.shape[2])

    def test_use_sides_for_recovery(self):
        reference_value = 30.19097
        adjustment = .25
        left_value = reference_value + adjustment
        left_test = adjust_side_images(reference_value, adjustment, 'left')
        self.assertEqual(left_value, left_test)
        right_value = reference_value - adjustment
        right_test = adjust_side_images(reference_value, adjustment, 'right')
        self.assertEqual(right_value, right_test)
        self.assertEqual(reference_value, adjust_side_images(reference_value, adjustment, 'center'))

    def test_shift_image_position(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        steering_angle = 30.67882
        test_translation_range = 0.004
        test_image, test_angle = shift_image_position(image, steering_angle, test_translation_range)
        self.assertNotAlmostEqual(steering_angle, test_angle)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], test_image.shape[0])
        self.assertEqual(image.shape[1], test_image.shape[1])
        self.assertEqual(image.shape[2], test_image.shape[2])

    def test_add_random_shadow(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        shadow_image = add_random_shadow(image)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], shadow_image.shape[0])
        self.assertEqual(image.shape[1], shadow_image.shape[1])
        self.assertEqual(image.shape[2], shadow_image.shape[2])

    def test_flip_image_and_measurement(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        measurement = 30.67882
        flipped_image, flipped_measurement = flip_image_and_measurement(image, measurement)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], flipped_image.shape[0])
        self.assertEqual(image.shape[1], flipped_image.shape[1])
        self.assertEqual(image.shape[2], flipped_image.shape[2])
        self.assertEqual(measurement*-1, flipped_measurement)

    def test_crop_image(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        horizon_divisor = 5
        hood_pixels = 25
        crop_height = 64
        crop_width = 64
        cropped_image = crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width)
        # Assert returned image's shape
        self.assertEqual(cropped_image.shape[0], crop_height)
        self.assertEqual(cropped_image.shape[1], crop_width)
        self.assertEqual(cropped_image.shape[2], 3)

    @unittest.skip
    def test_pick_random_vantage_point(self):
        # Line data goes in
        viewpoint = pick_random_vantage_point()
        # Hmmm. Just make sure it's not 'none', I guess.
        self.assertIn(viewpoint, ['left', 'right', 'center'])

    @unittest.skip
    def test_generate_train_by_batch(self):
        batch_size = 32
        # TODO: Figure out how to test the generator-- maybe a mock?

    def test_get_measurement_list(self):
        lines = [['center_2017_11_17_10_03_38_323.jpg',
                  'three_laps_counterclockwise/IMG/left_2017_11_17_10_03_38_323.jpg',
                  'right_2017_11_17_10_03_38_323.jpg',
                  '0','0','0','0.009133927'],
                 ['center_2017_11_17_10_03_38_416.jpg',
                  'left_2017_11_17_10_03_38_416.jpg',
                  'right_2017_11_17_10_03_38_416.jpg',
                  '0', '0', '0', '0.02888068'],
                 ['center_2017_11_17_10_03_38_512.jpg',
                  'left_2017_11_17_10_03_38_512.jpg',
                  'right_2017_11_17_10_03_38_512.jpg',
                  '0', '0', '0', '0.009574246']]
        test_measurements = [0.009133927, 0.02888068, 0.009574246]
        measurements = get_measurement_list(lines, 6)
        self.assertListEqual(measurements, test_measurements)

    def test_classify_measurements(self):
        measurements = [0.109133927, 0.02888068, -0.009574246, -0.102593028, 0]
        test_classes = [0, 1, 1, 0, 1]
        classes = classify_measurements(measurements)
        self.assertListEqual(classes, test_classes)

    def test_get_binary_downsample_indexes(self):
        measurements = [0.109133927, 0.02888068, -0.009574246, -0.102593028, 0]
        test_classes = [0, 1, 1, 0, 1]
        indexes = get_binary_downsample_indexes(measurements, test_classes)
        self.assertEqual(len(indexes), 4)

    def test_get_binary_downsample_ratio(self):
        classes = [0, 1, 1, 0, 1]
        test_ratio = get_binary_downsample_ratio(classes)
        check_ratio = {0: 2, 1:2}
        self.assertDictEqual(test_ratio, check_ratio)

    def test_binary_downsample_lines(self):
        lines = [['center_2017_11_17_10_03_38_323.jpg',
                  'three_laps_counterclockwise/IMG/left_2017_11_17_10_03_38_323.jpg',
                  'right_2017_11_17_10_03_38_323.jpg',
                  '0','0','0','0.109133927'],
                 ['center_2017_11_17_10_03_38_416.jpg',
                  'left_2017_11_17_10_03_38_416.jpg',
                  'right_2017_11_17_10_03_38_416.jpg',
                  '0', '0', '0', '0.02888068'],
                 ['center_2017_11_17_10_03_38_512.jpg',
                  'left_2017_11_17_10_03_38_512.jpg',
                  'right_2017_11_17_10_03_38_512.jpg',
                  '0', '0', '0', '-0.009574246'],
                 ['center_2017_11_17_10_03_38_512.jpg',
                  'left_2017_11_17_10_03_38_512.jpg',
                  'right_2017_11_17_10_03_38_512.jpg',
                  '0', '0', '0', '-0.102593028'],
                 ['center_2017_11_17_10_03_38_512.jpg',
                  'left_2017_11_17_10_03_38_512.jpg',
                  'right_2017_11_17_10_03_38_512.jpg',
                  '0', '0', '0', '0']]
        downsampled_lines = binary_downsample_lines(lines)
        self.assertLessEqual(len(downsampled_lines), len(lines))


if __name__ == '__main__':
    unittest.main()
