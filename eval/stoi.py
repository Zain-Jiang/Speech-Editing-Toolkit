from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import numpy as np
import warnings
import utils
import librosa
import torch

# Constant definition
FS = 22050  # Sampling frequency
N_FRAME = 1024  # Window support
NFFT = 1024  # FFT Size
OVERLAP = 4  # Number of steps to make in fftsize
MAX_WAV_VALUE = 32768.1

NUMBAND = 15  # Number of 13 octave band
MINFREQ = 150  # Center frequency of 1st octave band (Hz)
OBM, CF = utils.thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # Get 1/3 octave band matrix
N = 30  # N. frames for intermediate intelligibility
BETA = -15.  # Lower SDR bound
DYN_RANGE = 40  # Speech dynamic range


def stoi(x, y, fs_sig, extended=False):
    """ Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a clean
    signal, The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility.

    # Arguments
        x (np.ndarray): clean original speech
        y (np.ndarray): denoised speech
        fs_sig (int): sampling rate of x and y
        extended (bool): Boolean, whether to use the extended STOI

    # Returns
        float: Short time objective intelligibility measure between clean and
        denoised speech

    # Raises
        AssertionError : if x and y have different lengths
        
    """
    if x.shape != y.shape:
        raise Exception('x and y should have the same length,' +
                        'found {} and {}'.format(x.shape, y.shape))

    # Resample is fs_sig is different than fs
    if fs_sig != FS:
        x = utils.resample_oct(x, FS, fs_sig)
        y = utils.resample_oct(y, FS, fs_sig)

    # Remove silent frames
    if x.shape[0] < N_FRAME:
        return None
    x, y = utils.remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME / 2))

    # Take STFT
    x_spec = utils.stft(x, N_FRAME, NFFT, overlap=OVERLAP).transpose()
    y_spec = utils.stft(y, N_FRAME, NFFT, overlap=OVERLAP).transpose()

    # Ensure at least 30 frames for intermediate intelligibility
    if x_spec.shape[-1] < N:
        # warnings.warn('Not enough STFT frames to compute intermediate '
        #               'intelligibility measure after removing silent '
        #               'frames. Returning 1e-5. Please check you wav files',
        #               RuntimeWarning)
        return None

    # Apply OB matrix to the spectrograms as in Eq. (1)
    x_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec))))

    # Take segments of x_tob, y_tob
    x_segments = np.array(
        [x_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array(
        [y_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])

    if extended:
        x_n = utils.row_col_normalize(x_segments)
        y_n = utils.row_col_normalize(y_segments)
        return np.sum(x_n * y_n / N) / x_n.shape[0]

    else:
        # Find normalization constants and normalize
        normalization_consts = (
                np.linalg.norm(x_segments, axis=2, keepdims=True) /
                (np.linalg.norm(y_segments, axis=2, keepdims=True) + utils.EPS))
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-BETA / 20)
        y_primes = np.minimum(
            y_segments_normalized, x_segments * (1 + clip_value))

        # Subtract mean vectors
        y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms
        y_primes /= (np.linalg.norm(y_primes, axis=2, keepdims=True) + utils.EPS)
        x_segments /= (np.linalg.norm(x_segments, axis=2, keepdims=True) + utils.EPS)
        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments

        # J, M as in [1], eq.6
        J = x_segments.shape[0]
        M = x_segments.shape[1]

        # Find the mean of all correlations
        d = np.sum(correlations_components) / (J * M)
        return d


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sr = librosa.core.load(full_path, sr=FS)
    return torch.from_numpy(data).float(), sr


def cal_stoi(wav_pair):
    ref_path, gen_path = wav_pair
    ref_wav, _ = load_wav_to_torch(ref_path)
    gen_wav, _ = load_wav_to_torch(gen_path)
    min_length = min(ref_wav.size(0), gen_wav.size(0))
    ref_wav = ref_wav[:min_length]
    gen_wav = gen_wav[:min_length]
    ref_wav = ref_wav / MAX_WAV_VALUE
    gen_wav = gen_wav / MAX_WAV_VALUE

    ref = ref_wav.cpu().numpy()
    gen = gen_wav.cpu().numpy()
    score = stoi(ref, gen, FS, extended=False)
    valid = score is not None
    return score, valid

def cal_stoi_with_waves_batch(waves_dir):
    wavs = glob(waves_dir)
    
    wave_pairs = []
    for item in wavs:
        if '[G_SEG]' in item:
            wave_pairs.append((item, item.replace('G_SEG', 'P_SEG')))

    stoi_total, audio_num = 0, 0
    processes = 4
    with Pool(processes) as pool:
        for result_item in tqdm(pool.imap(cal_stoi, wave_pairs)):
            score, valid = result_item
            if valid:
                stoi_total += score
                audio_num += 1
    STOI_value = float(stoi_total) / float(audio_num)
    return STOI_value



if __name__ == '__main__':
    wavs = 'checkpoints/spec_denoiser_stutterset_dur_pitch_masked/generated_300000_/wavs/*'
    STOI_value = cal_stoi_with_waves_batch(wavs)
    print("STOI = {:f}".format(STOI_value))
