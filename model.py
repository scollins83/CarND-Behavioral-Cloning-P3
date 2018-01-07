import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
import sys
import argparse
import os
from keras.layers.convolutional import Convolution2D
from keras.optimizers import adam
import json
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
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
    Loads a .json configuration file
    :param config_name: Path for the configuration file.
    :return: A dict containing the loaded configuration values.
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration


def get_file_list(dir_path):
    """
    Get list of log files.
    :param dir_path: File path for .csv log file.
    :return: List of file paths of driving log files to open.
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
    Changes absolute file path name for an image noted in the log file to a new 'root'
    if directory was moved after recording.
    :param path: File path for an image.
    :param old_root: Original root to be replaced.
    :param new_root: New file root to be included.
    :return: Modified filepath.
    """
    file_tokens = path.split('/')
    end_tokens = file_tokens[-3:]
    end_tokens.insert(0, new_root)
    return '/'.join(end_tokens)


def get_image_and_measurement(line, old_root=None, new_root=None):
    """
    Gets images and measurements from a log file line.
    :param line: Line record from a log file.
    :param old_root: If changing image file paths, original root of the path.
    :param new_root: If changing image file paths, replacement root of the file path.
    :return: Center image, left image, right image, and steering measurement.
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
    Creates and compiles Convolutional Neural Network model to learn steering angle from images.
    Implemented from Bojarski, M. et al, NVIDIA, "End to End Learning for Self-Driving
    Cars", 2016 - https://arxiv.org/pdf/1604.07316v1.pdf, and this was noted by
    David Silver during the Udacity Self-Driving Car Nanodegree walkthrough for
    the behavioral cloning project.
    Before convolutional layers, the model normalizes and then crops the images.
    :param units: How many units to include in output layer (since this is a regression model,
    this value will always be 1.
    :param loss_function: Loss function to use in the model.
    :param input_shape: Input shape of images
    :param gpus: Number of GPUs to use, if training on a GPU with multiple cards.
    :param learning_rate: Learning rate to start with in the model.
    :param dropout: Percentage of records to dropout after convolutions.
    :return:
    """
    # NVIDIA Example
    conv_model = Sequential()
    conv_model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    conv_model.add(Cropping2D(cropping=((70,25),(0,0))))
    conv_model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
    conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
    conv_model.add(Dropout(dropout))
    conv_model.add(Flatten())
    conv_model.add(Dense(100, activation='relu'))
    conv_model.add(Dense(50, activation='relu'))
    conv_model.add(Dense(10, activation='relu'))
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


def augment_brightness_camera_images(image):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my Udacity mentor, Rahul. Function used to augment my dataset to
    improve model performance.
    :param image: Image file opened by OpenCV
    :return: Randomly brightened variation of the input image.
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
    as suggested reading by my Udacity mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param measurement_value: Steering measurement value.
    :param adjustment_offset: Amount to offset side images by.
    :param side: Image position to determine offset side.
    :return: Adjusted measurement value.
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
    which was a recommended reading by my Udacity mentor, Rahul. Function used to augment my dataset to
    improve model performance.
    :param image: Input image to shift.
    :param steering_angle: Input steering angle
    :param translation_range: Range to translate the image.
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
    which was a recommended reading by my Udacity mentor, Rahul. Function used to augment my dataset to
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
    as suggested reading by my Udacity mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param image: Image to be flipped
    :param measurement: Measurement to be flipped
    :return: Flipped image and measurement.
    """
    return cv2.flip(image, 1), measurement * -1


def crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width):
    """
    Note: Cited and refactored from blog post
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my Udacity mentor, Rahul. Function used to augment
    my dataset to improve model performance.
    :param image: Image to be cropped.
    :param horizon_divisor: Divisor to find the top 1/xth of the image to crop unneeded information above the horizon.
    :param hood_pixels: Number of pixels to crop out of the bottom of the image to eliminate unneeded information about the hood.
    :param crop_height: Height to resize the image to after eliminating horizon and hood.
    :param crop_width: Width to resize the image to after eliminating horizon and hood.
    :return: Cropped image.
    """
    shape = image.shape
    image = image[math.floor(shape[0] / horizon_divisor):shape[0] - hood_pixels, 0:shape[1]]
    image = cv2.resize(image, (crop_width, crop_height))

    return image


def full_augment_image(image, position, measurement, side_adjustment, shift_offset=0.004):
    """
    Takes an image array, it's position, it's measurement, the side adjustment value, and
    the translation offset desired and provides one copy of the original image plus all
    seven augmentation functions.
    :param image: Input image array.
    :param position: Position of the camera (Center, left or right)
    :param measurement: Steering angle measurement.
    :param shift_offset: Amount to offset translation.
    :return: Augmented images list, augmented measurements list. There should be 8.
    """
    aug_images = []
    aug_measurements = []

    measurement = adjust_side_images(measurement, side_adjustment, position)

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

    return aug_images, aug_measurements


def shuffle_lines(shuffle_lines):
    """
    Shuffles log record lines.
    :param shuffle_lines: Log record lines to be shuffled.
    :return: Shuffled lines
    """
    return_lines = []

    index_list = range(len(shuffle_lines))
    shuffled_indexes = random.sample(index_list, len(index_list))

    for i in shuffled_indexes:
        return_lines.append(shuffle_lines[i])

    return return_lines


def generate_full_augment_from_lines(lines, old_root, new_root, side_adjustment, batch_size=32):
    """
    Generator to generated full complement of augmented images from log line records.
    Batch size is divided by 8 to account for the 8 different images returned from
    the image augmentation function (assuming using 32, 64, 128, 256, etc. as acceptable
    batch size values).

    Splits lines into batches, opens images and gets measurement from the lines,
    augments each image (and updates the respective measurement accordingly),
    and converts the resulting lists of images and measurements to numpy arrays
    and yields them.
    :param lines: Log lines to augment.
    :param batch_size: Batch size for model run.
    :return: Augmented images and measurements yielded for model training.
    """
    num_lines = len(lines)
    batch_size = int(batch_size//8)
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


def get_measurement_list(lines, measurement_position=3):
    """
    Gets only the measurements from the lines.
    :param lines: Log record lines.
    :return: List of measurement values from the input log record lines.
    """
    measurement_list = [float(line[measurement_position]) for line in lines]
    logger.info("Measurement average: " + str(np.mean(measurement_list)))
    return measurement_list


def classify_measurements(measurements, low=-.1, high=.1):
    """
    Obtain class values based on the values of the input measurements.
    This is then used to downsample dominant value ranges.
    :param measurements: Steering angle measurements for the all log lines.
    :param low: Lower bound of class range.
    :param high: Upper bound of class range.
    :return: List of class for each measurement- 1 if within specified range,
    0 if not within that range.
    """
    classes = []
    for value in measurements:
        if value >= low and value <= high:
            classes.append(1)
        else:
            classes.append(0)
    logger.info("1 classes: " + str(len([x for x in classes if x == 1])))
    return classes


def get_binary_downsample_indexes(measurements, classes):
    """
    Gets a specified ratio and uses that to determine indexes
    to keep by undersampling the measurements according to class
    ratios. Undersampling utilizes the imbalanced-learn package's
    RandomUnderSampler.
    :param measurements: Steering angle measurements.
    :param classes: Classes of 1's or 0's pertaining to a dominant range.
    :return: Indexes for line records to keep as a result of undersampling.
    """
    ratio = get_binary_downsample_ratio(classes)
    logger.info(ratio)
    rus = RandomUnderSampler(ratio=ratio, return_indices=True)
    mmt_array = np.array(measurements).reshape(-1, 1)
    _, _, indexes = rus.fit_sample(mmt_array, classes)
    return indexes


def get_binary_downsample_ratio(classes, pct_keep_dominant=.9):
    """
    Calculates the downsampling ratio to pass to the imbalanced-learn
    RandomUnderSampler object. If 1's are not greater than 0's, return
    the ratio of the class counts as-is. If the number of 1's is greater
    than the number of zeros, return the full length of the zero class and
    multiply the 1's class by the percentage to keep of the dominant class.
    :param classes: List of zeros or ones labeling the measurements
    :return: Ratio dict to pass to the imbalanced-learrn RandomUnderSampler.
    """
    zeros = len([item for item in classes if item == 0])
    ones = len([item for item in classes if item == 1])
    if zeros >= ones:
        return {0: zeros, 1: ones}
    else:
        return {0: zeros, 1: int(ones*pct_keep_dominant)}


def binary_downsample_lines(lines):
    """
    Downsample dominant measurements by applying a binary indicator as to
    whether lines' particular measurements fall into a particular range, and
    then reducing records from the dominant class.
    :param lines: Log record lines
    :return: Log record lines, downsampled to remove some lines from the dominant range.
    """
    measurements = get_measurement_list(lines)
    classes = classify_measurements(measurements)
    logger.info('Len_Set_Classes: ' + str(len(set(classes))))
    if len(set(classes)) == 1:
        return lines
    else:
        indexes = get_binary_downsample_indexes(measurements, classes)
        return [lines[index] for index in indexes]


if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Load data
    logger.info("Getting log lines...")
    log_paths = get_file_list(config['input_path'])
    lines = []

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

    # Create and compile the model
    model = create_model(config['units'], gpus=config['gpus'], learning_rate=config['learning_rate'],
                         dropout=config['dropout_percentage'])

    # Set up checkpointing callback. Keep only the best checkpoints as the model improves.
    ckpt_path = config['checkpoint_path'] + "/augment_NVIDIA_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpointer = ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True)

    # Set up early stopping callback, to stop training if the validation loss hasn't
    # improved after 7 epochs.
    early_stopper = EarlyStopping(monitor='val_loss', patience=7)

    # Set up learning rate reducing callback, to decrease the learning rate
    # if the validation loss plateaus for 4 epochs.
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=4, min_lr=0.00000001)

    # Establish tensorboard callback, if configuration specified that Tensorboard should be used.
    # Add all callbacks to a list.
    if config["use_tensorboard"] == "True":
        if not os.path.exists(args['log_dir']):
            os.makedirs(args['log_dir'])
        tensorboard = TensorBoard(log_dir=args['log_dir'], histogram_freq=1, write_images=True)

        callbacks = [checkpointer, early_stopper, lr_reducer, tensorboard]
    else:
        callbacks = [checkpointer, early_stopper, lr_reducer]

    # Train the model, using training and validation generators to feed in images and measurements.
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

    # Save out the final results of the model.
    if config['output_path'].endswith('.h5'):
        model.save(config['output_path'])
    else:
        model.save(config['output_path'] + '.h5')

    # Clear the tensorflow session to try to surpress one minor warning that appears,
    # and end the program run.
    k.clear_session()
    sys.exit(0)
