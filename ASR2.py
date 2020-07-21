import python_speech_features as psf
import scipy.io.wavfile as sciwav
import os
import glob
import numpy as np
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
import random
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.activations import relu, sigmoid, softmax
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import regularizers

# ensuring repeatability of results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

PATH = "/home/marcin/Pobrane/TIMIT"


class TimitDatase():
    phonemes = ['h#', 'sh', 'ix', 'hv', 'eh', 'dcl', 'jh', 'ih', 'd', 'ah',
                'kcl', 'k', 's', 'ux', 'q', 'en', 'gcl', 'g', 'r', 'w',
                'ao', 'epi', 'dx', 'axr', 'l', 'y', 'uh', 'n', 'ae', 'm',
                'oy', 'ax', 'dh', 'tcl', 'iy', 'v', 'f', 't', 'pcl', 'ow',
                'hh', 'ch', 'bcl', 'b', 'aa', 'em', 'ng', 'ay', 'th', 'ax-h',
                'ey', 'p', 'aw', 'er', 'nx', 'z', 'el', 'uw', 'pau', 'zh',
                'eng', 'BLANK']

    def __init__(self, timit_root):
        self.max_label_len = 0

        # load the dataset
        training_root = os.path.join(timit_root, 'TRAIN')
        test_root = os.path.join(timit_root, 'TEST')

        self.ph_org_train, self.train_input_length, self.train_label_length, self.x_train, self.y_train = self.load_split_timit_data(
            training_root)
        self.ph_org_test, self.test_input_length, self.test_label_length, self.x_test, self.y_test = self.load_split_timit_data(
            test_root)
        self.normalize_xs()
        self.train_padded_ph = pad_sequences(self.y_train, maxlen=self.max_label_len, padding='post',
                                             value=len(self.phonemes))
        self.test_padded_ph = pad_sequences(self.y_test, maxlen=self.max_label_len, padding='post',
                                            value=len(self.phonemes))

    def num_classes(self):
        return len(self.phonemes)

    def normalize_xs(self):
        """
        Standarization 2D data
        """
        cut = int(self.x_train.shape[1] / 2)
        longX = self.x_train[:, -cut:, :]
        # flatten windows
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        # flatten train and test
        flatTrainX = self.x_train.reshape((self.x_train.shape[0] * self.x_train.shape[1], self.x_train.shape[2]))
        flatTestX = self.x_test.reshape((self.x_test.shape[0] * self.x_test.shape[1], self.x_test.shape[2]))
        # standardize
        s = StandardScaler()
        # fit on training data
        s.fit(longX)
        print("MEAN:")
        print(s.mean_)
        print("------------------------------------------")
        print("VAR:")
        print(s.var_)
        print("------------------------------------------")
        print("STD:")
        print(s.scale_)

        print(s.get_params(True))
        # apply to training and test data
        longX = s.transform(longX)
        flatTrainX = s.transform(flatTrainX)
        flatTestX = s.transform(flatTestX)
        # reshape
        self.x_train = flatTrainX.reshape((self.x_train.shape))
        self.x_test = flatTestX.reshape((self.x_test.shape))

    def filter_banks(self, signal, sample_rate):
        """
        Preprocessing data. If you would like to use MFCC, you should add DCT
        """
        pre_emphasis = 0.97
        frame_size = 0.025
        frame_stride = 0.01
        NFFT = 512
        nfilt = 40
        num_cep = 500
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z)  # Pad Signal

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]
        frames *= numpy.hamming(frame_length)
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB
        filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
        padding = np.zeros((num_cep, nfilt))
        padding[:filter_banks.shape[0], :filter_banks.shape[1]] = filter_banks[:num_cep, :]
        return padding

    def load_split_timit_data(self, root_dir):
        """
        Load and prepare TIMIT data to use ASR
        """
        wav_glob = os.path.join(root_dir, '**/SA1.wav')  # SA1 -> * (to read all DATA)

        x_list = []
        y_list = []
        ph_list = []
        label_length = []
        input_length = []
        for wav_filename in glob.glob(wav_glob, recursive=True):
            y_list_local = []
            ph_list_local = []
            sample_rate, wav = sciwav.read(wav_filename)
            filtered_data = self.filter_banks(wav, sample_rate)
            x_list.append(filtered_data)

            # parse the text file with phonemes
            phn_filename = wav_filename[:-3] + 'PHN'  # fragile, i know
            with open(phn_filename) as f:
                lines = f.readlines()
                phonemes = [line.split() for line in lines]

            for l, r, ph in phonemes:
                if len(x_list) % 100 == 0:
                    print('Added {} pairs.'.format(len(x_list)))

                phonem_idx = self.phonemes.index(ph)
                ph_list_local.append(ph)
                y_list_local.append(phonem_idx)
            y_list.append(y_list_local)
            ph_list.append((ph_list_local))
            label_length.append(len(y_list_local))
            input_length.append(100)
            if len(y_list_local) > self.max_label_len:
                self.max_label_len = len(y_list_local)

        x = np.array(x_list)
        y = np.array(y_list)
        ph_org_train = np.array(ph_list)
        train_input_length = np.array(input_length)
        train_label_length = np.array(label_length)
        return ph_org_train, train_input_length, train_label_length, x, y


class ASR():
    def __init__(self):
        """
        Create the model
        """
        inputs = Input(shape=(500, 40, 1))
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool_1 = AveragePooling2D(pool_size=(2, 2))(conv_1)
        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = AveragePooling2D(pool_size=(2, 2))(conv_2)
        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
        batch_norm_3 = BatchNormalization()(conv_3)
        pool_3 = AveragePooling2D(pool_size=(1, 2))(batch_norm_3)
        conv_4 = Conv2D(256, (2, 2), activation='relu', padding='same')(pool_3)
        batch_norm_4 = BatchNormalization()(conv_4)
        pool_4 = AveragePooling2D(pool_size=(1, 5))(batch_norm_4)
        lamb = Lambda(lambda x: K.squeeze(x, 2))(pool_4)
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(lamb)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(blstm_1)
        outputs = Dense(62, activation='softmax')(blstm_2)

        # model to be used at test time
        self.act_model = Model(inputs, outputs)
        self.act_model.summary()

        labels = Input(name='the_labels', shape=[data.max_label_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [outputs, labels, input_length, label_length])

        # model to be used at training time
        self.model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    def ctc_lambda_func(args, x):
        """
        Create cost function (CTC)
        """
        y_pred, labels, input_length, label_length = x

        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


if __name__ == "__main__":
    """
    Main function
    """

    data = TimitDatase(PATH)
    asr = ASR()

    asr.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    filepath = "best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    batch_size = 64
    epochs = 100
    asr.model.fit(x=[data.x_train, data.train_padded_ph, data.train_input_length, data.train_label_length],
                  y=np.zeros(len(data.x_train)), batch_size=batch_size, validation_split=0.1, epochs=epochs, verbose=1,
                  callbacks=callbacks_list, shuffle=True)

    # load the saved best model weights
    asr.act_model.load_weights('best_model.hdf5')

    # predict outputs on validation images
    prediction = asr.act_model.predict(data.x_test[:10])

    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])

    # see the results
    i = 0
    for x in out:
        print("original_text =  ", data.ph_org_test[i])
        print("predicted text = ", end='')
        for p in x:
            if int(p) != -1:
                print("'" + data.phonemes[int(p)] + "', ", end='')
        print('\n')
        i += 1
