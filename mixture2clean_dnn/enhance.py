import os
import time
import glob
import pickle
import argparse
import numpy as np

from keras.models import load_model

import config as cfg
from audio_utils import read_audio, calculate_spectrogram
from system_utils import create_directory
import audio_utils
from spectrogram_to_wave import recover_wav


def extract_features(workspace, speech_to_enhance_dir, snr):
    time_start = time.time()

    sample_rate = cfg.sample_rate

    audios_to_enhance_dir = os.path.join(workspace, speech_to_enhance_dir)
    for audio_id, audio_path in enumerate(glob.glob(audios_to_enhance_dir + "/*.wav")):
        speech_audio = read_audio(audio_path, target_fs=sample_rate)[0]

        speech_audio_complex_spectrogram = calculate_spectrogram(speech_audio, mode="complex",
                                                                 window_size=cfg.n_window, n_overlap=cfg.n_overlap)

        # Save features.
        features = [speech_audio_complex_spectrogram, os.path.basename(audio_path).split(".")[0]]

        features_filename = "{}.pickle".format(os.path.basename(audio_path).split(".")[0])
        features_dir = os.path.join(workspace, "data", "speech_to_enhance", "features", "spectrogram", "{}db".format(int(snr)))
        create_directory(features_dir)

        features_path = os.path.join(features_dir, features_filename)
        pickle.dump(features, open(features_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("Extracting features time: %s" % (time.time() - time_start))
    print()


def enhance_audio(workspace, speech_to_enhance_dir, train_snr, test_snr, n_concat, iteration):
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
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "{}db".format(int(train_snr)), "scaler.pickle")
    scaler = pickle.load(open(scaler_path, "rb"))

    # Load test data.
    features_dir = os.path.join(workspace, "data", "speech_to_enhance", "features", "spectrogram", "{}db".format(int(test_snr)))

    for sample_id, feature_filename in enumerate(os.listdir(features_dir)):
        # Load feature.
        feature_path = os.path.join(features_dir, feature_filename)
        feature_data = pickle.load(open(feature_path, "rb"))

        mixed_audio_complex_spectrogram, audio_name = feature_data
        mixed_audio_spectrogram = np.abs(mixed_audio_complex_spectrogram)

        # Process data.
        n_pad = (n_concat - 1) / 2
        mixed_audio_spectrogram = audio_utils.pad_with_border(mixed_audio_spectrogram, n_pad)
        mixed_audio_spectrogram = audio_utils.log_sp(mixed_audio_spectrogram)

        # Scale data.
        if scale:
            mixed_audio_spectrogram = audio_utils.scale_on_2d(mixed_audio_spectrogram, scaler)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_audio_spectrogram_3d = audio_utils.mat_2d_to_3d(mixed_audio_spectrogram, agg_num=n_concat, hop=1)

        # Predict.
        prediction = model.predict(mixed_audio_spectrogram_3d)
        print("Sample id: {}. sample name: {}".format(sample_id, audio_name))

        # Inverse scale.
        if scale:
            prediction = audio_utils.inverse_scale_on_2d(prediction, scaler)

        # Recover enhanced wav.
        prediction_spectrogram = np.exp(prediction)
        recovered_wave = recover_wav(prediction_spectrogram, mixed_audio_complex_spectrogram, n_overlap, np.hamming)

        # Scaler for compensate the amplitude change after spectrogram and IFFT.
        recovered_wave *= np.sqrt((np.hamming(n_window) ** 2).sum())

        # Write out enhanced wav.
        enhanced_audio_filename = "{}.enh.wav".format(audio_name)
        enhanced_audio_dir = os.path.join(workspace, "data", "speech_to_enhance", "enhanced_wavs", "{}db".format(int(test_snr)))

        create_directory(enhanced_audio_dir)
        audio_utils.write_audio(os.path.join(enhanced_audio_dir, enhanced_audio_filename), recovered_wave, fs)

    print()
    print("Inference time: {}".format(time.time() - begin_time))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    parser_enhance_audio = subparsers.add_parser("enhance_audio")
    parser_enhance_audio.add_argument("--workspace", type=str, required=True)
    parser_enhance_audio.add_argument('--speech_to_enhance', type=str, required=True)

    args = parser.parse_args()

    if args.mode == "enhance":
        enhance(args)
