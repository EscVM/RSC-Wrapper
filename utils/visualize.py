# Copyright 2021 PIC4SeR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cv2
from scipy.stats import logistic


def plotImages(images_batch, img_n, classes):
    """
    Take as input a batch from the generator and plt a number of images equal to img_n
    Default columns equal to max_c. At least inputs of batch equal two.
    """
    max_c = 5
    
    if img_n <= max_c:
        r = 1
        c = img_n
    else:
        r = math.ceil(img_n/max_c)
        c = max_c
        
    fig, axes = plt.subplots(r, c, figsize=(15,15))
    axes = axes.flatten()
    for img_batch, label_batch, ax in zip(images_batch[0], images_batch[1], axes):
        ax.imshow(img_batch)
        ax.grid()
        ax.set_title('Class: {}'.format(classes[label_batch]))
    plt.tight_layout()
    plt.show()


def plot_misclassified_images(X_test, y_pred, y_test, labels):
    """
    Plot all misclassified images.
    """
    y_pred_arg = (logistic.cdf(y_pred) > 0.5)[...,0].astype(np.float32)
    errors_indices = np.where(y_pred_arg != y_test)[0]
    
    max_c = 5
    
    r = np.ceil(errors_indices.size/max_c).astype(np.int32)
    c = max_c
        
    fig, axes = plt.subplots(r, c, figsize=(55,55))
    axes = axes.flatten()
    for e_index, ax in zip(errors_indices, axes):
        ax.imshow(X_test[e_index].astype(np.uint8))
        ax.grid()
        class_pred = logistic.cdf(y_pred[int(e_index)])[0]
        if class_pred < 0.5:
            class_pred = 1 - class_pred
        ax.set_title('Class {}: {:.2%}'.format(labels[int(y_pred_arg[int(e_index)])], 
                                            class_pred), color='red')
    plt.tight_layout()
    plt.show()
