"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""

import os
import csv
import glob
import time
import h5py
import pickle
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler

import config as cfg
from audio_utils import read_audio, write_audio, get_amplitude_scaling_factor, additive_mixing, \
    calculate_spectrogram, pad_with_border, mat_2d_to_3d, log_sp

from system_utils import create_directory


def create_rules_for_mixing_speech_with_noises(workspace, speech_dir, noise_dir, data_type, magnification):
    """Create csv containing mixture information.

    Each row in the .csv file contains [speech_filename, noise_filename, noise_onset, noise_offset]

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      magnification: int, only used when data_type='train', number of noise
          selected to mix with a speech. E.g., when magnification=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger
          than the species of noises.
    """
    time_start = time.time()

    random_state = np.random.RandomState(42)

    rules_dir = os.path.join(workspace, "mixing_rules")
    create_directory(rules_dir)

    rules_filename = os.path.join(rules_dir, "{}.csv".format(data_type))
    with open(rules_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speech_file_name", "noise_file_name", "noise_begin", "noise_end"])

        noise_paths = glob.glob(noise_dir + "*.wav")
        speech_paths = glob.glob(speech_dir + "*.wav")

        for speech_path in speech_paths:
            (speech_audio, _) = read_audio(speech_path)

            # For training data, mix each speech with randomly picked #magnification noises.
            # For test data, mix each speech with all noises.
            if data_type == "train":
                noise_paths = random_state.choice(noise_paths, size=magnification, replace=False)

            for noise_path in noise_paths:
                (noise_audio, _) = read_audio(noise_path)

                if noise_audio.shape[0] <= speech_audio.shape[0]:
                    noise_begin = 0
                    noise_end = noise_audio.shape[0]
                else:
                    # If noise longer than speech then randomly select a segment of noise.
                    noise_begin = random_state.randint(0, noise_audio.shape[0] - speech_audio.shape[0], size=1)[0]
                    noise_end = noise_begin + speech_audio.shape[0]

                writer.writerow([os.path.basename(speech_path), os.path.basename(noise_path), noise_begin, noise_end])

    print()
    print("Mixing clean {} speech with noises time: {}".format(data_type, time.time() - time_start))
    print()


def calculate_mixture_features(workspace, speech_dir, noise_dir, data_type, snr):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the
    features to disk.

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
    """
    time_start = time.time()

    fs = cfg.sample_rate

    # Open mixture csv.
    rules_filename = os.path.join(workspace, "mixing_rules", "{}.csv".format(data_type))
    with open(rules_filename, "r", encoding="utf-8") as f:
        rules_reader = csv.reader(f)
        next(rules_reader, None)  # skip the headers

        for i, rule in enumerate(rules_reader):
            [speech_filename, noise_filename, noise_begin, noise_end] = rule

            speech_path = os.path.join(speech_dir, speech_filename)
            speech_audio = read_audio(speech_path, target_fs=fs)[0]

            noise_path = os.path.join(noise_dir, noise_filename)
            noise_audio = read_audio(noise_path, target_fs=fs)[0]

            # Repeat noise n_repeat times to cover entire clean speech sample.
            if noise_audio.shape[0] < speech_audio.shape[0]:
                n_repeat = int(np.ceil(speech_audio.shape[0] / noise_audio.shape[0]))
                noise_audio = np.tile(noise_audio, n_repeat)[:speech_audio.shape[0]]
            # Truncate noise to the same length as speech.
            else:
                noise_audio = noise_audio[int(noise_begin):int(noise_end)]

            # Scale speech to given SNR.
            scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
            speech_audio *= scaler

            # Get normalized mixture, speech, noise.
            mixed_audio, speech_audio, noise_audio, alpha = additive_mixing(speech_audio, noise_audio)

            rule_name = "{}.{}".format(speech_filename.split(".")[0], noise_filename.split(".")[0])

            # Save mixed audio.
            mixed_audio_filename = "{}.wav".format(rule_name)
            mixed_audio_dir = os.path.join(workspace, "mixed_audios", "spectrogram", data_type, "{}db".format(int(snr)))
            create_directory(mixed_audio_dir)

            write_audio(os.path.join(mixed_audio_dir, mixed_audio_filename), mixed_audio, fs)

            # Extract spectrograms.
            mixed_audio_complex_spectrogram = calculate_spectrogram(mixed_audio, mode='complex',
                                                                    window_size=cfg.n_window, n_overlap=cfg.n_overlap)
            speech_spectrogram = calculate_spectrogram(speech_audio, mode='magnitude',
                                                       window_size=cfg.n_window, n_overlap=cfg.n_overlap)
            noise_spectrogram = calculate_spectrogram(noise_audio, mode='magnitude',
                                                      window_size=cfg.n_window, n_overlap=cfg.n_overlap)

            # Save features.
            features_filename = "{}.{}.pickle".format(speech_filename.split(".")[0], noise_filename.split(".")[0])
            features_dir = os.path.join(workspace, "features", "spectrogram", data_type, "{}db".format(int(snr)))
            create_directory(features_dir)

            features = [mixed_audio_complex_spectrogram, speech_spectrogram, noise_spectrogram, alpha, rule_name]
            feature_path = os.path.join(features_dir, features_filename)
            pickle.dump(features, open(feature_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

            if (i + 1) % 101 == 0:
                print("Iteration # {}".format(i))

    print()
    print("Extracting features time: %s" % (time.time() - time_start))
    print()


def pack_features(workspace, data_type, snr, n_concat, n_hop):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file.

    Args:
      workspace: str, path of workspace.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
      n_concat: int, number of frames to be concatenated.
      n_hop: int, hop frames.
    """

    inputs = []  # (n_segs, n_concat, n_freq)
    outputs = []  # (n_segs, n_freq)

    time_begin = time.time()

    # Load all features.
    features_dir = os.path.join(workspace, "features", "spectrogram", data_type, "{}db".format(int(snr)))
    for i, features_path in enumerate(glob.glob(features_dir + "/*.pickle")):
        # Load feature.
        data = pickle.load(open(features_path, "rb"))
        mixed_audio_complex_spectrogram, speech_spectrogram, noise_spectrogram, alpha, rule_name = data

        mixed_audio_complex_spectrogram = np.abs(mixed_audio_complex_spectrogram)

        # Pad start and finish of the spectrogram with boarder values.
        n_pad = (n_concat - 1) / 2
        mixed_audio_complex_spectrogram = pad_with_border(mixed_audio_complex_spectrogram, n_pad)
        speech_spectrogram = pad_with_border(speech_spectrogram, n_pad)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_audio_complex_spectrogram_3d = mat_2d_to_3d(mixed_audio_complex_spectrogram, agg_num=n_concat, hop=n_hop)
        inputs.append(mixed_audio_complex_spectrogram_3d)

        # Cut target spectrogram and take the center frame of each 3D segment.
        speech_spectrogram_3d = mat_2d_to_3d(speech_spectrogram, agg_num=n_concat, hop=n_hop)
        y = speech_spectrogram_3d[:, int((n_concat - 1) / 2), :]
        outputs.append(y)

        if (i + 1) % 101 == 0:
            print("Iteration # {}".format(i))

    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    inputs = log_sp(inputs).astype(np.float32)
    outputs = log_sp(outputs).astype(np.float32)

    # Write out data to .h5 file.
    features_filename = "data.h5"
    features_dir = os.path.join(workspace, "packed_features", "spectrogram", data_type, "{}db".format(int(snr)))
    create_directory(features_dir)

    with h5py.File(os.path.join(features_dir, features_filename), "w") as hf:
        hf.create_dataset('x', data=inputs)
        hf.create_dataset('y', data=outputs)

    print()
    print("Packing features time: {}".format(time.time() - time_begin))
    print()


def compute_scaler(workspace, data_type, snr):
    """Compute and write out scaler of data.
    """

    # Load data.
    begin_time = time.time()

    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "{}db".format(int(snr)), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x')) # (n_segs, n_concat, n_freq)

    # Compute scaler.
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = StandardScaler(with_mean=True, with_std=True).fit(x2d)

    # Write out scaler.
    scaler_filename = "scaler.pickle"
    scaler_dir = os.path.join(workspace, "packed_features", "spectrogram", data_type, "{}db".format(int(snr)))
    create_directory(scaler_dir)

    pickle.dump(scaler, open(os.path.join(scaler_dir, scaler_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("Building scaler time: {}".format(time.time() - begin_time))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixing_rules')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)
    
    args = parser.parse_args()
    if args.mode == 'create_mixing_rules':
        workspace = args.workspace
        speech_dir = args.speech_dir
        noise_dir = args.noise_dir
        data_type = args.data_type
        magnification = args.magnification

        create_rules_for_mixing_speech_with_noises(workspace, speech_dir, noise_dir, data_type, magnification)

    elif args.mode == 'calculate_mixture_features':
        workspace = args.workspace
        speech_dir = args.speech_dir
        noise_dir = args.noise_dir
        data_type = args.data_type
        snr = float(args.snr)

        calculate_mixture_features(workspace, speech_dir, noise_dir, data_type, snr)

    elif args.mode == 'pack_features':
        workspace = args.workspace
        data_type = args.data_type
        snr = float(args.snr)
        n_concat = int(args.n_concat)
        n_hop = int(args.n_hop)

        pack_features(workspace, data_type, snr, n_concat, n_hop)

    elif args.mode == 'compute_scaler':
        workspace = args.workspace
        data_type = args.data_type
        snr = args.snr

        compute_scaler(workspace, data_type, snr)
    else:
        raise Exception("Error!")