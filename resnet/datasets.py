import torch as tc
import torchvision as tv
import numpy as np


class CIFAR10HePreprocessing(tc.utils.data.Dataset):
    # CIFAR-10 with preprocessing as described in Section 4.2 of He et al., 2015.
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.dataset = tv.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None, target_transform=None)
        #self.per_pixel_means = self.get_per_pixel_means()
        self.per_channel_means = self.get_per_channel_means()
        self.transform = None
        self.target_transform = None

    def get_per_pixel_means(self):
        training_data = tv.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=None)

        X, y = training_data[0]
        X = np.array(X)
        per_pixel_means = np.zeros(dtype=np.float32, shape=X.shape)

        for i in range(0, len(training_data)):
            X, y = training_data[i]
            X = np.array(X).astype(np.float32)
            X = X / 255. # convert to [0, 1] range
            X = 2.0 * X - 1.0 # convert to [-1, 1] range
            per_pixel_means += X

        per_pixel_means = per_pixel_means / float(len(training_data))
        return per_pixel_means

    def get_per_channel_means(self):
        training_data = tv.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=None)

        X, y = training_data[0]
        X = np.array(X)
        per_channel_means = np.zeros(dtype=np.float32, shape=(1, 1, 3))

        for i in range(0, len(training_data)):
            X, y = training_data[i]
            X = np.array(X).astype(np.float32)
            X = X / 255. # convert to [0, 1] range
            X = 2.0 * X - 1.0 # convert to [-1, 1] range
            per_channel_means += np.mean(X, axis=(0,1), keepdims=True) # per-channel mean, in NHWC format.

        per_channel_means = per_channel_means / float(len(training_data))
        return per_channel_means

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        X = np.array(X).astype(np.float32)
        X = X / 255.  # convert to [0, 1] range
        X = 2.0 * X - 1.0  # convert to [-1, 1] range

        ## subtract per-pixel mean following He et al., 2015
        image = X - self.per_channel_means
        #image = (X - self.per_pixel_means)
        #image = X
        image = np.transpose(image, [2,0,1]) # NCHW

        label = y
        if self.train:
            ## pad image.
            h, w = image.shape[1:]
            pad_pixels = 4
            vertical_padding = np.zeros(dtype=np.float32, shape=(3, 2*pad_pixels+h, pad_pixels))
            horizontal_padding = np.zeros(dtype=np.float32, shape=(3, pad_pixels, w))
            image = np.concatenate([vertical_padding, np.concatenate([horizontal_padding, image, horizontal_padding], axis=1), vertical_padding], axis=2)

            ## randomly flip image horizontally.
            flip_prob = 0.50
            if np.random.uniform(0.0, 1.0) > flip_prob:
                image = image[:, :, ::-1]

            ## randomly crop image.
            padded_image_h, padded_image_w = h+2*pad_pixels, w+2*pad_pixels
            h_start = np.random.randint(low=0, high=padded_image_h-h) ## assumes we want a crop of size [h,w].
            h_end = h_start + h
            w_start = np.random.randint(low=0, high=padded_image_w-w)
            w_end = w_start + w
            image = image[:, h_start:h_end, w_start:w_end]
            image = image.copy()

        return image, label
