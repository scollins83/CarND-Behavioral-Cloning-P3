import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
import sys
import argparse
import os
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
import json
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import time
import logging
import copy
import math
import random
from keras import backend as k
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tfl = logging.getLogger('tensorflow')
tfl.setLevel(logging.ERROR)


def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-c', '--configuration', help="File path configuration file", required=True,
                        dest='config')
    parser.add_argument('-l', '--logdir', help="Tensorboard log directory", dest='log_dir', required=True)

    return vars(parser.parse_args(arguments))


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration


def get_file_list(dir_path):
    """
    Get list of files
    :param dir_path:
    :return: List of driving log files to open.
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        [file_list.append(os.path.join(root, file)) for file in files if file.endswith('.csv')]
    return file_list


def get_log_lines(path):
    """
    Gets list of records from driving log.
    :param path: Input path for driving log.
    :return: List of driving log records.
    """
    log_lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            log_lines.append(line)
    return log_lines


def get_path_replacement(path, new_root):
    """

    :param path:
    :param old_root:
    :param new_root:
    :return:
    """
    file_tokens = path.split('/')
    end_tokens = file_tokens[-3:]
    end_tokens.insert(0, new_root)
    return '/'.join(end_tokens)


def get_image_and_measurement(line, old_root=None, new_root=None):
    """

    :param line:
    :param old_root:
    :param new_root:
    :return:
    """
    center_image_path = line[0]
    left_image_path = line[1]
    right_image_path = line[2]

    if new_root and old_root:
        center_image_path = get_path_replacement(center_image_path, new_root)
        left_image_path = get_path_replacement(left_image_path, new_root)
        right_image_path = get_path_replacement(right_image_path, new_root)
    center_image = cv2.imread(center_image_path)
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    measurement = float(line[3])

    return center_image, left_image, right_image, measurement


def create_model(units=1, loss_function='mse', input_shape=(160, 320, 3),
                 gpus=1, learning_rate=0.001, dropout=0.25):
    """
    Constructs Keras model object
    :return: Compiled Keras model object
    """
    # NVIDIA Example
    conv_model = Sequential()
    conv_model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    conv_model.add(Cropping2D(cropping=((70,25),(0,0))))
    #conv_model.add(Convolution2D(3, 1, 1, input_shape=input_shape))
    conv_model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
    conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
    conv_model.add(Dropout(dropout))
    conv_model.add(Flatten())
    #conv_model.add(Dense(1164))
    conv_model.add(Dense(100))
    conv_model.add(Dense(50))
    conv_model.add(Dense(10))
    conv_model.add(Dense(units))

    opt = adam(lr=learning_rate)

    if gpus <= 1:
        conv_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
    else:
        gpu_list = []
        [gpu_list.append('gpu(%d)' % i) for i in range(gpus)]
        conv_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'],
                           context=gpu_list)
    return conv_model


def custom_get_params(self):
    """
    Function to patch issue in Keras
    :param self: Sci-kit parameters.
    :return: Deep copy of the parameters.
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res


def augment_brightness_camera_images(image):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image: Image file opened by OpenCV
    :return:
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def adjust_side_images(measurement_value, adjustment_offset, side):
    """
    Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav,
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    as suggested reading by my mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param measurement_value:
    :param adjustment_offset:
    :param side:
    :return:
    """
    if side == 'left':
        return measurement_value + adjustment_offset
    elif side == 'right':
        return measurement_value - adjustment_offset
    elif side == 'center':
        return measurement_value


def shift_image_position(image, steering_angle, translation_range):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image:
    :param steering_angle:
    :param translation_range:
    :return: translated_image, translated_steering_angle
    """
    translation_x = translation_range * np.random.uniform() - translation_range / 2
    translated_steering_angle = steering_angle + translation_x / translation_range * 2 * .2
    translation_y = 40 * np.random.uniform() - 40 / 2
    translation_m = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rows = image.shape[0]
    cols = image.shape[1]
    translated_image = cv2.warpAffine(image, translation_m, (cols, rows))

    return translated_image, translated_steering_angle


def add_random_shadow(image):
    """
    Adding a random shadow mask to the image.
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image: Image to add a shadow too.
    :return: Image with a random shadow added.
    """
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    x_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((x_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


def flip_image_and_measurement(image, measurement):
    """
    Flips image so it looks as though it was made going from the opposite direction.

    Note: Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav,
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    as suggested reading by my mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param image:
    :param measurement:
    :return:
    """
    return cv2.flip(image, 1), measurement * -1


def crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width):
    """
    Note: Cited and refactored from blog post
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image:
    :param horizon_divisor:
    :param hood_pixels:
    :param crop_height:
    :param crop_width:
    :return:
    """
    shape = image.shape
    image = image[math.floor(shape[0] / horizon_divisor):shape[0] - hood_pixels, 0:shape[1]]
    image = cv2.resize(image, (crop_width, crop_height))

    return image


def full_augment_image(image, position, measurement, side_adjustment, shift_offset=0.004):
    """

    :param image:
    :param position:
    :param measurement:
    :param shift_offset:
    :return:
    """
    aug_images = []
    aug_measurements = []

    measurement = adjust_side_images(measurement, .25, position)

    bright = augment_brightness_camera_images(image)
    shadow = add_random_shadow(image)
    flipped, flipped_mmt = flip_image_and_measurement(image, measurement)

    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(bright)
    aug_measurements.append(measurement)
    aug_images.append(shadow)
    aug_measurements.append(measurement)
    aug_images.append(flipped)
    aug_measurements.append(flipped_mmt)

    if position == 'center':
        translated, shift_mmt = shift_image_position(image, measurement, shift_offset)
        bright_shifted, bright_shift_mmt = shift_image_position(bright, measurement, shift_offset)
        shadow_shifted, shadow_shift_mmt = shift_image_position(shadow, measurement, shift_offset)
        flipped_shifted, flipped_shift_mmt = shift_image_position(flipped, flipped_mmt, shift_offset)

        aug_images.append(translated)
        aug_measurements.append(shift_mmt)

        aug_images.append(bright_shifted)
        aug_measurements.append(bright_shift_mmt)

        aug_images.append(shadow_shifted)
        aug_measurements.append(shadow_shift_mmt)

        aug_images.append(flipped_shifted)
        aug_measurements.append(flipped_shift_mmt)

    # So, for non-center, will have 4, and if center, will have 8. So, 12 images per row.
    return aug_images, aug_measurements


def shuffle_lines(shuffle_lines):
    """

    :param shuffle_lines:
    :return: shuffled_lines
    """
    return_lines = []

    index_list = range(len(shuffle_lines))
    shuffled_indexes = random.sample(index_list, len(index_list))

    for i in shuffled_indexes:
        return_lines.append(shuffle_lines[i])

    return return_lines


def generate_from_lines(lines, old_root, new_root, side_adjustment, batch_size=32):
    """

    :param lines:
    :param batch_size:
    :return:
    """
    num_lines = len(lines)
    while 1:
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_line in batch_lines:
                center, left, right, measurement = get_image_and_measurement(batch_line,
                                                                             old_root,
                                                                             new_root)
                images.append(center)
                images.append(left)
                images.append(right)

                measurements.append(measurement)
                measurements.append(measurement + side_adjustment)
                measurements.append(measurement - side_adjustment)

            # Add a flipped version for each image.
                f_img, f_mmt = flip_image_and_measurement(center, measurement)
                images.append(f_img)
                measurements.append(f_mmt)

                f_img, f_mmt = flip_image_and_measurement(left, (measurement + side_adjustment))
                images.append(f_img)
                measurements.append(f_mmt)


                f_img, f_mmt = flip_image_and_measurement(right, (measurement - side_adjustment))
                images.append(f_img)
                measurements.append(f_mmt)

            X = np.array(images)
            y = np.array(measurements)

            yield shuffle(X, y)


def generate_full_augment_from_lines(lines, old_root, new_root, side_adjustment, batch_size=32):
    """

    :param lines:
    :param batch_size:
    :return:
    """
    num_lines = len(lines)
    while 1:
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_line in batch_lines:
                center, left, right, measurement = get_image_and_measurement(batch_line,
                                                                             old_root,
                                                                             new_root)

                for position in ['center','left','right']:
                    if position == 'center':
                        current_image = center
                    elif position == 'left':
                        current_image = left
                    else:
                        current_image = right

                    augmented_images, augmented_measurements = full_augment_image(current_image,
                                                                                  position,
                                                                                  measurement,
                                                                                  side_adjustment)

                    for index in range(len(augmented_measurements)):
                        images.append(augmented_images[index])
                        measurements.append(augmented_measurements[index])

            X = np.array(images)
            y = np.array(measurements)

            yield shuffle(X, y)


def get_measurement_list(lines, measurement_position=6):
    """
    Gets just the measurements from the lines.
    :param lines:
    :return:
    """
    return [float(line[measurement_position]) for line in lines]


def classify_measurements(measurements, low=-.1, high=.1):
    """

    :param measurements:
    :param low:
    :param high:
    :return:
    """
    classes = []
    for value in measurements:
        if value >= low and value <= high:
            classes.append(1)
        else:
            classes.append(0)
    return classes


def get_binary_downsample_indexes(measurements, classes):
    """

    :param measurements:
    :param classes:
    :return:
    """
    ratio = get_binary_downsample_ratio(classes)
    #logger.info(ratio)
    rus = RandomUnderSampler(ratio=ratio, return_indices=True)
    mmt_array = np.array(measurements).reshape(-1, 1)
    _, _, indexes = rus.fit_sample(mmt_array, classes)
    return indexes


def get_binary_downsample_ratio(classes, pct_keep_dominant=.8):
    """

    :param classes:
    :return:
    """
    zeros = len([item for item in classes if item == 0])
    ones = len([item for item in classes if item == 1])
    if zeros >= ones:
        return {0: zeros, 1: ones}
    else:
        return {0: zeros, 1: int(ones*pct_keep_dominant)}


def binary_downsample_lines(lines):
    """

    :param lines:
    :return:
    """
    measurements = get_measurement_list(lines)
    classes = classify_measurements(measurements)
    if len(set(classes)) == 1:
        return lines
    else:
        #logger.info([print(str(measurements[i]) + ' : ' + str(classes[i])) for i in range(len(measurements))])
        indexes = get_binary_downsample_indexes(measurements, classes)
        return [lines[index] for index in indexes]


if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Load data
    # lines = get_log_lines(config['input_path'])
    logger.info("Getting log lines...")
    log_paths = get_file_list(config['input_path'])
    lines = []

    #[lines.append([path, get_log_lines(path)]) for path in log_paths]
    for path in log_paths:
        [lines.append(line) for line in get_log_lines(path)]
    logger.info("Number of lines: " + str(len(lines)))

    # Downsample the lines to better balance out the 0-value measurements
    lines = binary_downsample_lines(lines)
    logger.info("Number of lines after downsampling: " + str(len(lines)))

    # Shuffle the data once
    lines = shuffle_lines(lines)
    logger.info("Number of lines after shuffle: " + str(len(lines)))


    # Train/Test Split
    validation_index = int(len(lines) * config['test_size'])

    lines_test = lines[-validation_index:]
    lines_train = lines[:-validation_index]
    logger.info("Validation set of length " + str(len(lines_test)))
    logger.info("Training set of length " + str(len(lines_train)))

    model = create_model(config['units'], gpus=config['gpus'], learning_rate=config['learning_rate'],
                         dropout=config['dropout_percentage'])
    ckpt_path = config['checkpoint_path'] + "/augment_NVIDIA_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpointer = ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True)

    # Establish tensorboard
    if config["use_tensorboard"] == "True":
        tensorboard = TensorBoard(log_dir=args['log_dir'], histogram_freq=1,
        #tensorboard = TensorBoard(log_dir=config['tensorboard_log_dir'], histogram_freq=1,
                                  write_graph=True)
        callbacks = [checkpointer, tensorboard]
    else:
        callbacks = [checkpointer]

    logger.info("Training the model...")

    train_generator = generate_full_augment_from_lines(lines_train, config['old_image_root'],
                                          config['new_image_root'],
                                          config['side_adjustment'], config['batch_size'])

    validation_generator = generate_full_augment_from_lines(lines_test, config['old_image_root'],
                                               config['new_image_root'],
                                               config['side_adjustment'], config['batch_size'])

    model.fit_generator(train_generator, samples_per_epoch=len(lines_train), nb_epoch=config['epochs'],
                        validation_data=validation_generator, nb_val_samples=len(lines_test), callbacks=callbacks,
                        nb_worker=3, nb_val_worker=2)


    if config['output_path'].endswith('.h5'):
        model.save(config['output_path'])
    else:
        model.save(config['output_path'] + '.h5')

    k.clear_session()
    sys.exit(0)
