#!/usr/bin/python3

import numpy as np

def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
        ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`my_conv2d_freq` function in `part4.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def my_sharpen_freq(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the input image using a sharpening filter.
    
    Return the sharpened image.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, k)
    Returns:
        sharpened_image: array of shape (m, n)
    HINTS:
    1. Apply sharpening filter to the input image using my_conv2d_freq()
        - Hint: You can use a 3x3 Laplacian filter for the sharpening
    2. Normalize the obtained sharpened image.
    3. Enchance image by adding image obtained from sharpening filter to the original image to obtain the final sharpened image
    - You should use the my_conv2d_freq function to help you with this task.
    """

    ############################
        ### TODO: YOUR CODE HERE ###

    raise NotImplementedError(
        "`my_sharpen_freq` function in `part4.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image