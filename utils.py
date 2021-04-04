import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.fft as nf
from scipy.signal import find_peaks


def plot_learning_curve(history):
    """
    Plot the learning curve.

    :param history: Keras API model training history
    :return: None
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def to_cat(data, num_classes=None):
    """
    Change the label to One-hot coding.

    :param data: label to change as in [1, 2, 3, 4]
    :param num_classes: total numbers of classes
    :return: Encoded label
    """
    # Each data should be represents the class number of the data
    if num_classes is None:
        num_classes = np.unique(data)
    data_class = np.zeros((data.shape[0], num_classes))
    for i in range(data_class.shape[0]):
        num = data[i]
        data_class[i, num] = 1
    return data_class


def translate_params(params):
    """
    Translate the list of parameters to the corresponding parameter(CNN).

    :param params: list, [dropout, learning_rate, batch_size, number of convolution]
    :return: value of dropout(float), learning_rate(float) and batch_size(int)
    """
    conv_candidate = [4, 6, 8]
    bs_candidate = [1, 16, 32, 64, 128, 256]
    dropout = params[0]
    learning_rate = params[1]
    num_batch_size = int(params[2])
    num_conv = int(params[3])
    batch_size = bs_candidate[num_batch_size]
    conv = conv_candidate[num_conv]
    assert conv in conv_candidate
    assert batch_size in bs_candidate

    return dropout, learning_rate, batch_size, conv


def translate_params_svm(params):
    """
    Translate the list of parameters to the corresponding parameter(SVM).

    :param params: list, [C, kernel option]
    :return: c, kernel function
    """
    c_candidate = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    kernel_candidate = ['rbf', 'linear', 'poly', 'sigmoid']
    num_c = int(params[0])
    num_kernel = int(params[1])
    c = c_candidate[num_c]
    kernel = kernel_candidate[num_kernel]
    assert c in c_candidate
    assert kernel in kernel_candidate

    return c, kernel


def print_params(params, net=None):
    """
    print the cnn params via translating function

    :param net:
    :param params: list of cnn parameters
    :return: None
    """
    if net == "CNN":
        dropout, learning_rate, batch_size, conv = translate_params(params)
        print('Best parameters: ',
              '\ndropout=', dropout,
              'learning rate=', learning_rate,
              'batch size=', batch_size,
              'number of convolution=', conv)
    elif net == "DBN":
        print('Best parameters: ',
              '\ndropout=', params[0],
              'LearningRate_rbm=', params[1],
              'LearningRate_nn=', params[2])
    elif net == "SVM":
        C, kernel = translate_params_svm(params)
        print("Best Parameters: ",
              "\nC=", C,
              "Kernel function:", kernel)


def data_FFT(data):
    data_fft = []
    for i in range(len(data)):
        rank_i = data[i]
        # print(rank1)
        times = np.arange(rank_i.size)
        freqs = nf.fftfreq(times.size, times[1] - times[0])
        xs = np.abs(freqs)
        complex_array = nf.fft(rank_i)
        ys = np.abs(complex_array)
        # ## plot signal in time domain
        # plt.figure()
        # plt.plot(times, rank_i)
        # plt.title("Signal[0] in Time Domain")
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")
        # plt.show()
        # ## plot signal in frequency domain
        # plt.figure()
        # plt.plot(xs, ys)
        # plt.xlabel("Frequency")
        # plt.title('Frequency Domain', fontsize=16)
        # plt.ylabel('Amplitude', fontsize=12)
        # plt.tick_params(labelsize=10)
        # plt.grid(linestyle=':')
        # plt.show()
        ## find peaks in frequency domain
        peak_id, peak_property = find_peaks(ys, height=6, distance=10)
        peak_freq = xs[peak_id]
        peak_height = peak_property['peak_heights']
        peak_freq = np.unique(peak_freq)
        if peak_freq is not None:
            peak_freq = np.append(peak_freq[0], peak_freq[-1])  # select first and last peaks
        peak_height = np.unique(peak_height)
        if peak_height is not None:
            peak_height = np.append(peak_height[0], peak_height[-1])
        else:
            print("peak_freq not found, change params")
        # print('peak_freq',peak_freq)
        # print('peak_height',peak_height)
        data_i_fft = np.append(peak_freq, peak_height)
        # print(data_i_fft)
        data_fft.append(data_i_fft)
    data_fft = np.asarray(data_fft).reshape(len(data), 4)  # generate new x_train from frequency domain
    # print(data_fft)
    return data_fft



