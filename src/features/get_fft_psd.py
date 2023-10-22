import numpy as np
from scipy.signal import welch
from scipy.fftpack import fftn


def get_fft_values(df: np.ndarray,
                   T: float,
                   N: int,
                   f_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Get Fourier coefficients and frequencies"""
    # df: np.ndarray["n_samples", "ts_len", "n_features"]
    # since we are only interested in the magnitude of the amplitudes, 
    # we use np.abs() to take the real part of the frequency spectrum.
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values = fftn(df, axes=(1, 2))
    fft_values = 2.0 / N * np.abs(fft_values[:, 0:N // 2, :])
    return f_values, fft_values

def get_psd_values(df: np.ndarray,
                   T: float,
                   N: int,
                   f_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Get Power Spectral Density and frequencies"""
    # df: np.ndarray["n_samples", "ts_len", "n_features"]
    features = []
    for i in range(df.shape[-1]):
        f_values, psd_values = welch(df[:,:,i], fs=f_s)
        features.append(np.abs(psd_values))
    features = np.moveaxis(np.array(features), [0, 1, 2], [2, 0, 1])
    return f_values, features
