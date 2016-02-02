import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter
import sys
import os


def train_vae(font_h5, model, save_path='./model_vae/default', n_epochs=100, read_size=5000, batch_size=50, save_freq=5):

    data_size = font_h5.shape[0]
    epp = data_size/read_size
    for j in range(n_epochs*epp):
        start = np.random.randint(data_size-read_size)
        q = font_h5[start:start+read_size].astype('f')
        q /= 255
        gf(q)
        q = np.random.permutation(q)
        model.fit(q, n_epochs=1, save_freq=-1, pic_freq=-1, model_path='./', img_path='./', batch_size=100)
        sys.stdout.flush()
        status = j/epp
        print("Epoch %d" % status)
        if j % save_freq == 0:

            sample = np.random.standard_normal((1, model.latent_width)).astype('f')
            out = model.inverse_transform(sample, test=True).transpose(0, 3, 1, 2)
            out = (1.-out)
            plt.figure(figsize=(8, 8))
            gs1 = gridspec.GridSpec(8, 8)
            gs1.update(wspace=0.0, hspace=0.0)
            for i, letter in enumerate(out[0]):
                plt.subplot(8, 8, i+1)
                ax1 = plt.subplot(gs1[i])
                ax1.imshow(letter, 'gray')
                plt.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_aspect('equal')
            plt.show()
            model.save(os.path.dirname(save_path), os.path.basename(save_path))


def train_gan(font_h5, model, letter='A', save_path='./model_gan/default', n_epochs=100, read_size=5000, batch_size=50, save_freq=5):
    cap_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z']
    low_list = map(lambda x: x.lower(), cap_list)
    num = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']
    space = [' ']
    chars = cap_list + low_list + num + space
    mapper = {name: i for i, name in enumerate(chars)}

    char_number = mapper[letter]
    data_size = font_h5.shape[0]
    epp = data_size/read_size
    for j in range(n_epochs*epp):

        start = np.random.randint(data_size-read_size)
        q = font_h5[start:start+read_size][:, char_number]
        q = q[:, np.newaxis, :, :].astype('f')
        q /= 255
        gf(q)
        q = np.random.permutation(q)
        model.fit(q.astype('f'), n_epochs=1, save_freq=-1, pic_freq=-1, model_path='./', img_path='./', batch_size=100)
        sys.stdout.flush()
        status = j/epp
        print("Epoch %d" % status)
        if j % save_freq == 0:
            sample = np.random.standard_normal((10, model.latent_width)).astype('f')
            out = model.inverse_transform(sample, test=True).transpose(0, 3, 1, 2)
            out = 1. - out
            plt.figure(figsize=(15, 1))
            for i in range(10):
                plt.subplot(1, 10, i+1)
                plt.imshow(out[i, 0], 'gray')
                plt.axis("off")
            plt.show()
            model.save(os.path.dirname(save_path), os.path.basename(save_path))


def interpolating_font(model, text, stride=10):

    cap_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z']
    low_list = map(lambda x: x.lower(), cap_list)
    num = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']
    space = [' ']
    chars = cap_list + low_list + num + space
    mapper = {name: i for i, name in enumerate(chars)}
    length = len(text)
    row = length/20 + (length % 20 != 0)
    code = [mapper[text[i]] for i in range(length)]

    n_vecs = length/stride + (length % stride != 0) + 1

    vecs = np.random.standard_normal((n_vecs, model.latent_width)).astype('f')

    dirs = vecs[1:] - vecs[:-1]

    dirs /= stride

    interp = []
    for direction, vec in zip(dirs, vecs[:-1]):
        seq = [(vec + direction*i).tolist() for i in range(stride)]
        interp += seq
    interp = np.array(interp).astype('f')

    blank = np.ones((64, 64))
    blank[0, 0] = 0

    plt.figure(figsize=(20, row))
    gs1 = gridspec.GridSpec(row, 20)
    gs1.update(wspace=0.0, hspace=0.0)
    out = model.inverse_transform(interp, test=True).transpose(0, 3, 1, 2)
    for i, j in enumerate(code):

        plt.subplot(row, 20, i+1)
        ax1 = plt.subplot(gs1[i])
        if j != 62:
            ax1.imshow(1.-out[i, j], 'gray')
        else:
            ax1.imshow(blank, 'gray')
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    plt.show()


def show_latent_2d(model):
    w = model.latent_width
    assert w == 2, \
        'Latent dimension %d != 2. Please retrain to have dimension 2.' % w
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.array([[y0, x0] for x0 in x for y0 in y], dtype='f')

    fig = plt.figure(figsize=(20, 10))
    out = model.inverse_transform(z, test=True).transpose(0, 3, 1, 2)
    out = 1. - out
    for i in range(200):
        plt.subplot(10, 20, i+1)
        plt.imshow(out[i, 0], 'gray')
        plt.axis("off")
    plt.show()


def gf(fonts_batch, sigma=1.):
    for i in range(fonts_batch.shape[0]):
        fonts_batch[i, :, :, :] = \
            gaussian_filter(fonts_batch[i, :, :, :].transpose(1, 2, 0),
                            sigma=(sigma, sigma, 0)).transpose(2, 0, 1)
