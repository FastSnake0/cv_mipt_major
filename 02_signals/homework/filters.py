import numpy as np
import cv2
from scipy.signal import convolve2d

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops. 💀💀💀
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
    # Выполняем свертку с использованием 4 вложенных циклов
    for i in range(Hi):
        for j in range(Wi):

            summ = 0
            for m in range(Hk):
                for n in range(Wk):
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
    # Используем np.pad для добавления нулевых отступов
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
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
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE

    # Flip the kernel
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    Hk, Wk = kernel.shape
    delta_h = int((Hk - 1) / 2)
    delta_w = int((Wk - 1) / 2)
    for image_h in range(delta_h, Hi-delta_h):
        for image_w in range(delta_w, Wi-delta_w):
            out[image_h][image_w] = np.sum(kernel*image[image_h-delta_h:image_h+delta_h+1,image_w-delta_w:image_w+delta_w+1])
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    An efficient implementation of convolution filter using scipy's convolve2d.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    return out
    

def cross_correlation(f, g):
    """Cross-correlation of f and g with resizing using OpenCV.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    # Определяем размер f
    Hf, Wf = f.shape

    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]

    flipped_g = np.flip(np.flip(g, axis=(0, 1)))

    # Используем conv_fast для выполнения свёртки с перевёрнутым ядром
    out = conv_fast(f, flipped_g)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    # Вычитаем среднее значение ядра g
    g_zero_mean = g - np.mean(g)

    # Определяем размеры
    Hf, Wf = f.shape
    Hk, Wk = g.shape
    pad_height = Hk // 2
    pad_width = Wk // 2

    # Паддинг для изображения
    padded_image = np.pad(f, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Результирующий массив
    out = np.zeros((Hf, Wf))

    # Выполняем кросс-корреляцию
    for i in range(Hf):
        for j in range(Wf):
            # Извлекаем текущую область изображения
            region = padded_image[i:i + Hk, j:j + Wk]
            # Вычисляем корреляцию с ядром с нулевым средним
            out[i, j] = np.sum(region * g_zero_mean)

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

    out = None

    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]

    Hk, Wk = g.shape
    Hi, Wi = f.shape
    out = np.zeros((Hi, Wi))

    normalized_filter = (g - np.mean(g)) / np.std(g)
    assert np.mean(normalized_filter) < 1e-5, "g mean is {}, should be 0".format(np.mean(g))
    assert np.abs(np.std(normalized_filter) - 1) < 1e-5, "g std is {}, should be 1".format(np.std(g))

    delta_h = int((Hk - 1) / 2)
    delta_w = int((Wk - 1) / 2)
    for image_h in range(delta_h, Hi - delta_h):
        for image_w in range(delta_w, Wi - delta_w):
            image_patch =f[image_h - delta_h:image_h + delta_h + 1, image_w - delta_w:image_w + delta_w + 1]
            normalized_image_patch = (image_patch - np.mean(image_patch)) / np.std(image_patch)
            out[image_h][image_w] = np.sum(normalized_filter * normalized_image_patch)

    return out