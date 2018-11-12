"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""

import numpy as np
import os
import pickle
import h5py
import argparse
import time
import glob

import config as cfg
from data_generator import DataGenerator
from spectrogram_to_wave import recover_wav

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model

import audio_utils
from system_utils import load_hdf5


def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = audio_utils.np_mean_absolute_error(y_all, pred_all)
    return loss


def save_training_stats(iteration, tr_loss, te_loss, stats_dir):
    stat_dict = {'iter': iteration, 'tr_loss': tr_loss, 'te_loss': te_loss}
    stat_path = os.path.join(stats_dir, "{}iters.pickle".format(iteration))
    pickle.dump(stat_dict, open(stat_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def train(args):
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    
    # Load data. 
    t1 = time.time()

    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")

    (tr_x, tr_y) = load_hdf5(tr_hdf5_path)
    (te_x, te_y) = load_hdf5(te_hdf5_path)

    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)

    print("Load data time: %s s" % (time.time() - t1,))
    
    batch_size = 500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    
    # Scale data. 
    if True:
        t1 = time.time()

        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.pickle")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = audio_utils.scale_on_3d(tr_x, scaler)
        tr_y = audio_utils.scale_on_2d(tr_y, scaler)
        te_x = audio_utils.scale_on_3d(te_x, scaler)
        te_y = audio_utils.scale_on_2d(te_y, scaler)

        print("Scale data time: %s s" % (time.time() - t1,))

    # Build model
    (_, n_concat, n_freq) = tr_x.shape
    n_hid = 2048
    
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    model.summary()
    
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=lr))

    # Data generator. 
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)
    
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)
    
    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    
    # Save out training stats. 
    save_training_stats(iter, tr_loss, te_loss, stats_dir)

    # Train. 
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1
        
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            
            # Save out training stats. 
            save_training_stats(iter, tr_loss, te_loss, stats_dir)

        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        
        if iter == 10001:
            break
            
    print("Training time: %s s" % (time.time() - t1,))


def inference(workspace, train_snr, test_snr, n_concat, iteration, visualize):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace.
      train_snr: float, training SNR.
      test_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iteration: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    begin_time = time.time()

    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True
    
    # Load model. 
    model_path = os.path.join(workspace, "models", "{}db".format(int(train_snr)), "md_{}iters.h5".format(iteration))
    model = load_model(model_path)
    
    # Load scaler. 
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "{}db".format(int(train_snr)),
                               "scaler.pickle")
    scaler = pickle.load(open(scaler_path, "rb"))
    
    # Load test data. 
    features_dir = os.path.join(workspace, "features", "spectrogram", "test", "{}db".format(int(test_snr)))

    for sample_id, feature_filename in enumerate(os.listdir(features_dir)):
        # Load feature. 
        feature_path = os.path.join(features_dir, feature_filename)
        feature_data = pickle.load(open(feature_path, "rb"))

        mixed_audio_complex_spectrogram, speech_spectrogram, noise_spectrogram, alpha, rule_name = feature_data
        mixed_audio_spectrogram = np.abs(mixed_audio_complex_spectrogram)
        
        # Process data. 
        n_pad = (n_concat - 1) / 2
        mixed_audio_spectrogram = audio_utils.pad_with_border(mixed_audio_spectrogram, n_pad)
        mixed_audio_spectrogram = audio_utils.log_sp(mixed_audio_spectrogram)
        speech_spectrogram = audio_utils.log_sp(speech_spectrogram)
        
        # Scale data. 
        if scale:
            mixed_audio_spectrogram = audio_utils.scale_on_2d(mixed_audio_spectrogram, scaler)
            speech_spectrogram = audio_utils.scale_on_2d(speech_spectrogram, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat.
        mixed_audio_spectrogram_3d = audio_utils.mat_2d_to_3d(mixed_audio_spectrogram, agg_num=n_concat, hop=1)
        
        # Predict. 
        prediction = model.predict(mixed_audio_spectrogram_3d)
        print("Sample id: {}. sample name: {}".format(sample_id, rule_name))

        # Inverse scale.
        if scale:
            mixed_speech_spectrogram = audio_utils.inverse_scale_on_2d(mixed_audio_spectrogram, scaler)
            speech_spectrogram = audio_utils.inverse_scale_on_2d(speech_spectrogram, scaler)
            prediction = audio_utils.inverse_scale_on_2d(prediction, scaler)

        # Recover enhanced wav. 
        prediction_spectrogram = np.exp(prediction)
        recovered_wave = recover_wav(prediction_spectrogram, mixed_audio_complex_spectrogram, n_overlap, np.hamming)

        # Scaler for compensate the amplitude change after spectrogram and IFFT.
        recovered_wave *= np.sqrt((np.hamming(n_window)**2).sum())

        # Write out enhanced wav.
        enhanced_audio_filename = "{}.enh.wav".format(rule_name)
        enhanced_audio_dir = os.path.join(workspace, "enhanced_wavs", "test", "{}db".format(int(test_snr)))

        audio_utils.create_directory(enhanced_audio_dir)
        audio_utils.write_audio(os.path.join(enhanced_audio_dir, enhanced_audio_filename), recovered_wave, fs)

    print()
    print("Inference time: {}".format(time.time() - begin_time))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference':
        workspace = args.workspace
        train_snr = args.tr_snr
        test_snr = args.te_snr
        n_concat = args.n_concat
        iteration = args.iteration
        visualize = args.visualize

        inference(workspace, train_snr, test_snr, n_concat, iteration, visualize)

    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")
