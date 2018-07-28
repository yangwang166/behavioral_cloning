import cv2
import csv
import numpy as np
import os
import sklearn
import warnings
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split

warnings.filterwarnings("always")

def prepare_data(csv_path):
    folder_name = csv_path.split("/")[0]
    center_imgs = []
    left_imgs = []
    right_imgs = []
    steer_angles = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            #line: center, left, right, steering angle, throttle, break, speed
            center_imgs.append(folder_name + "/" + line[0].strip())
            left_imgs.append(folder_name + "/" + line[1].strip())
            right_imgs.append(folder_name + "/" + line[2].strip())
            steer_angles.append(float(line[3].strip()))
    return (center_imgs, left_imgs, right_imgs, steer_angles)

csv_path_1 = 'new_data/driving_log.csv'
center_paths_1, left_paths_1, right_paths_1, steer_angles_1 = prepare_data(csv_path_1)

csv_path_2 = 'new_data_reverse/driving_log.csv'
center_paths_2, left_paths_2, right_paths_2, steer_angles_2 = prepare_data(csv_path_2)

all_paths = []
all_paths.extend(center_paths_1)
all_paths.extend(left_paths_1)
all_paths.extend(right_paths_1)
all_paths.extend(center_paths_2)
all_paths.extend(left_paths_2)
all_paths.extend(right_paths_2)

all_angles = []
correction = 0.2
all_angles.extend(steer_angles_1)
all_angles.extend([angle + correction for angle in steer_angles_1])
all_angles.extend([angle - correction for angle in steer_angles_1])
all_angles.extend(steer_angles_2)
all_angles.extend([angle + correction for angle in steer_angles_2])
all_angles.extend([angle - correction for angle in steer_angles_2])

samples = list(zip(all_paths, all_angles))

def nvidia_car_model():
    """
    Implementation of nvidia autonomous car model
    """

    # Prepreocessing layers
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping = ((50, 20), (0,0))))

    # Main network
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

model = nvidia_car_model()

model.compile(loss = 'mse', optimizer = 'adam')

def my_generator(samples, batch_size = 32):
    """
    Generate the required images and angles for training `samples` is a list of pairs (`image_path`, `angle`).
    """

    # Total number of samples
    num_samples = len(samples)

    # Using yeild to output batch result, until gothough all data in samples
    while True:

        # Random shuflling of Samples
        samples = sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []

            for image_path, angle in batch_samples:
                # Get image
                original_image = cv2.imread(image_path)

                # Convert color space from BGR to RGB
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                images.append(image)
                angles.append(angle)

                # Data augmentation: flipping, add more data
                images.append(cv2.flip(image, 1))
                angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = my_generator(train_samples)
validation_generator = my_generator(validation_samples)

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch = len(train_samples),
                                     validation_data = validation_generator,
                                     nb_val_samples = len(validation_samples),
                                     nb_epoch = 3,
                                     verbose = 1)


model.save('model_new.h5')

print("Loss", history_object.history['loss'])
print("Validation Loss", history_object.history['val_loss'])
