import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input # change with given backbone

INPUT_SIZE =  224 # default pacs image dimension

def horizontal_flip(x, y):
    """
    Random horizontal flip of an image.
    """
    x = tf.image.random_flip_left_right(x)
    return x, y


def grayscale(x, y):
    """
    Image to gray with 10% probability.
    """
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    if choice < 0.1:
        gray_tensor = tf.image.rgb_to_grayscale(x)
        return tf.repeat(gray_tensor, repeats=3, axis=-1), y
    else:
        return x, y


def random_crop(x, y):
    """
    Random crop between 80% and 100%.
    """
    perc = tf.math.floor(tf.random.uniform(shape=[], minval=0.8, maxval=1., dtype=tf.float32)*224)
    image_crops = tf.image.random_crop(x, [perc, perc, 3])
    return tf.image.resize(image_crops, [INPUT_SIZE,INPUT_SIZE]), y

def standardize(x, y):
    """
    Standardize data with imagenet means
    """
    return preprocess_input(x), y
