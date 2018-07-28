# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: http://chuantu.biz/t6/349/1532793242x-1404829175.png "Model Summary"
[image2]: http://chuantu.biz/t6/349/1532794089x-1404829175.jpg "Center"
[image3]: http://chuantu.biz/t6/349/1532795040x-1404829175.jpg "Recovery Image 1"
[image4]: http://chuantu.biz/t6/349/1532795061x-1404829175.jpg "Recovery Image 2"
[image5]: http://chuantu.biz/t6/349/1532795078x-1404829175.jpg "Recovery Image 3"
[image6]: http://chuantu.biz/t6/349/1532795514x-1404829175.png "Steering Angle Distribution"
[image7]: http://chuantu.biz/t6/349/1532795709x-1404829175.png "Original Image"
[image8]: http://chuantu.biz/t6/349/1532795744x-1404829175.png "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: Containing the script to create and train the model
* drive.py: For driving the car in autonomous mode
* model.h5: Containing a trained convolution neural network
* writeup.md: Summarizing the results
* model.h5: The final model file
* video.py: Generating the video
* output.mp4: The record video of autonomous car in Simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing

Since I use another GPU server to train the model, and that Server have the proper environment for training. However that that GPU server don't have graphic interface. So after training the model, I have to copy the `model.h5` back to my macbook to work with Simulator. If you also train and run on two different machine, please make sure both of them have proper environment to run the code, `model.py` and `driver.py`. You may also need to run `conda env create -f environment.yml` on your local environment.

```sh
python drive.py model.h5
```

I also increase the car speed to 20 in `driver.py`

```
set_speed = 20
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have tried LeNet at the very beginning, because I have use it before in Project2 and familiar with it. But this network cannot provide a good model. Then I changed to Nvidia's network. And that network after trained 1 epochs, already can produced a good enough model.

Here is the model summary:

![alt text][image1]

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (`model.py`)

```python
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
```

The model includes RELU layers to introduce nonlinearity (above), and the data is normalized in the model using a Keras lambda layer:

```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
```


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting:

```python
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.2))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting:

```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually:

```python
model.compile(loss = 'mse', optimizer = 'adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy Details

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use Nvidia autonomous car model as a base. And I add some dropout layer and change the size of convolutional layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layer after each full connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I add additional training data, such as reverse track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

```python
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
    model.add(Dropout(0.6))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model
```


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to back to center. These images show what a recovery looks like starting from right sides back to center :

![Recover 1][image3]

![Recover 2][image4]

![Recover 3][image5]

Then I repeated this process on track two in order to get more data points.

##### Here is some key points:

* The official data seems using keyboard to control the car, the steering angle distribution are very unbalanced.
* Then I change to use mouse to control the steering angle, which can give me a much better angle distribution.
* There are 3 camera on the car, in order to fully utilize the 3 camera, I combined 3 camera's data together, and manually calcuate the steering angle for the left and right camera:

``` python
all_paths = []
#all_paths.extend(center_paths_0)
#all_paths.extend(left_paths_0)
#all_paths.extend(right_paths_0)
all_paths.extend(center_paths_1)
all_paths.extend(left_paths_1)
all_paths.extend(right_paths_1)
all_paths.extend(center_paths_2)
all_paths.extend(left_paths_2)
all_paths.extend(right_paths_2)

all_angles = []
correction = 0.2
#all_angles.extend(steer_angles_0)
#all_angles.extend([angle + correction for angle in steer_angles_0])
#all_angles.extend([angle - correction for angle in steer_angles_0])
all_angles.extend(steer_angles_1)
all_angles.extend([angle + correction for angle in steer_angles_1])
all_angles.extend([angle - correction for angle in steer_angles_1])
all_angles.extend(steer_angles_2)
all_angles.extend([angle + correction for angle in steer_angles_2])
all_angles.extend([angle - correction for angle in steer_angles_2])
```

* After merge all these images, the final steering angle like this:

![Steering Angle Distribution][image6]


To augment the data sat, I also flipped images and angles thinking that this would add more data. For example, here is an image that has then been flipped:

![Original][image7]

![Flipped][image8]


After the collection process, I had `27102` number of data points.

I finally randomly shuffled the data set and put `20%` of the data into a validation set.

Also I am using generator to save memory, and Keras Lambda Layer to crop the image to focus on the meaningful area.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2, since I can find the `loss` stop decreasing from epochs 2 (loss is 0.0034 ). I used an `adam optimizer` so that manually training the learning rate wasn't necessary.

### Output video

When the speed is 10, the autonomous car can run pretty good in Simulator.
Then I increase speed to 20, and recorded the video as follow:

* First Person View: https://youtu.be/c2sOITDiteA
* Third Person View: https://youtu.be/xkVOUPdo57c
