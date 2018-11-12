import librosa
import soundfile
import numpy as np

from scipy import signal


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if (target_fs is not None) and (fs != target_fs):
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def rms(y):
    """Root mean square.
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_amplitude_scaling_factor(speech, noise, snr):
    """Given speech and noise, return the scaler, according to the SNR.

    Args:
      speech: ndarray, source1.
      noise: ndarray, source2.
      snr: float, SNR.

    Outputs:
      float, scaler.
    """
    original_sn_rms_ratio = rms(speech) / rms(noise)
    target_sn_rms_ratio = 10. ** (float(snr) / 20.)  # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio

    return signal_scaling_factor


def additive_mixing(signal, noise):
    """Mix normalized source1 and source2.

    Args:
      signal: ndarray, source1.
      noise: ndarray, source2.

    Returns:
      mix_audio: ndarray, mixed audio.
      s: ndarray, pad or truncated and scaled source1.
      n: ndarray, scaled source2.
      alpha: float, normalize coefficient.
    """
    mixed_audio = signal + noise

    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha

    signal *= alpha
    noise *= alpha

    return mixed_audio, signal, noise, alpha


def calculate_spectrogram(audio, mode, window_size, n_overlap):
    """Calculate spectrogram.

    Args:
      audio: 1darray.
      mode: string, 'magnitude' | 'complex'
      window_size: int, windows size for FFT.
      n_overlap: int, overlap of window.
    Returns:
      spectrogram: 2darray, (n_time, n_freq).
    """

    haming_window = np.hamming(window_size)
    [f, t, x] = signal.spectral.spectrogram(
        audio,
        window=haming_window,
        nperseg=window_size,
        noverlap=n_overlap,
        detrend=False,
        return_onesided=True,
        mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * int(n_pad) + [x] + [x[-1:]] * int(n_pad)
    return np.concatenate(x_pad_list, axis=0)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    len_x, n_in = x.shape
    if len_x < agg_num:
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def log_sp(x):
    return np.log(x + 1e-08)


def scale_on_2d(x2d, scaler):
    """Scale 2D array data.
    """
    return scaler.transform(x2d)


def scale_on_3d(x3d, scaler):
    """Scale 3D array data.
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))

    return x3d


def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data.
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]


def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
