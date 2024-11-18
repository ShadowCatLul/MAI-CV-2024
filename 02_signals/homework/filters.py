import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    kernel_flipped = np.flip(np.flip(kernel, 0), 1)

    # Iterate over each pixel in the output
    for i in range(Hi):
        for j in range(Wi):
            # Initialize the sum for the current position
            summ = 0
            # Iterate over the kernel
            for m in range(Hk):
                for n in range(Wk):
                    # Calculate the position in the image
                    if (i + m - Hk // 2) >= 0 and (i + m - Hk // 2) < Hi and (j + n - Wk // 2) >= 0 and (j + n - Wk // 2) < Wi:
                        summ += image[i + m - Hk // 2, j + n - Wk // 2] * kernel_flipped[m, n]
            # Assign the computed sum to the output pixel
            out[i, j] = summ


    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    
    # Place the original image in the center of the padded array
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_height = Hk // 2
    pad_width = Wk // 2
    # Pad the image
    padded_image = zero_pad(image, pad_height, pad_width)

    # Flip the kernel
    kernel_flipped = np.flip(np.flip(kernel, 0), 1)

    out = np.zeros((Hi, Wi))

    for i in range(Hi):
            for j in range(Wi):
                # Extract the region of interest
                region = padded_image[i:i + Hk, j:j + Wk]
                # Perform element-wise multiplication and sum the result
                out[i, j] = np.sum(region * kernel_flipped)

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_height = Hk // 2
    pad_width = Wk // 2
    padded_image = zero_pad(image, pad_height, pad_width)
    kernel_flipped = np.flip(kernel)
    out = np.zeros((Hi, Wi))
    for i in range(Hi):
            for j in range(Wi):
                # Extract the region of interest
                region = padded_image[i:i + Hk, j:j + Wk]
                # Perform element-wise multiplication and sum the result
                out[i, j] = np.sum(region * kernel_flipped)
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    pad_height = Hk // 2
    pad_width = Wk // 2

    padded_image = zero_pad(f, pad_height, pad_width)

    kernel_sum_of_squares = np.sum(g ** 2)

    out = np.zeros((Hi, Wi))

    for i in range(Hi):
            for j in range(Wi):
                # Extract the region of interest
                region = padded_image[i:i + Hk, j:j + Wk]
                # Perform element-wise multiplication and sum the result
                norm_coeff = np.sqrt(kernel_sum_of_squares * np.sum(region ** 2))
                
                out[i, j] = np.sum(region * g) / norm_coeff if norm_coeff != 0 else 0

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """


    g_zero_mean = g - np.mean(g)

    out = cross_correlation(f, g_zero_mean)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    f_height, f_width = f.shape
    g_height, g_width = g.shape
    
    # Initialize the output array
    out = np.zeros((f_height - g_height + 1, f_width - g_width + 1))
    
    # Calculate mean and standard deviation of the template
    g_mean = np.mean(g)
    g_std = np.std(g)
    g_norm = (g - g_mean) / g_std
    g_sum_sq = np.sum(g ** 2)
    
    # Iterate over each position in the output
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            # Extract the patch from the image
            f_patch = f[i:i+g_height, j:j+g_width]
            
            # Calculate mean and standard deviation of the patch
            f_patch_mean = np.mean(f_patch)
            f_patch_std = np.std(f_patch)
            f_path_sum_sq= np.sum(f_patch**2)


            out[i, j] = np.sum((f_patch - f_patch_mean) * g_norm / f_patch_std) / np.sqrt(f_path_sum_sq*g_sum_sq)


    return out
