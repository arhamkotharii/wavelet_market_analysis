import numpy as np
import pywt

SUPPORTED_WAVELETS = [
    "haar",
    "db4",
    "sym4",
    "coif1"
]

SUPPORTED_TRANSFORMS = [
    "dwt",
    "swt"
]

# ============================================================
# HAAR DWT
# ============================================================

def haar_dwt(signal, levels):

    current = np.asarray(
        signal,
        dtype=float
    )

    details = []

    for _ in range(levels):

        if len(current) % 2 != 0:
            current = current[:-1]

        low = (
            current[0::2]
            + current[1::2]
        ) / np.sqrt(2)

        high = (
            current[0::2]
            - current[1::2]
        ) / np.sqrt(2)

        details.append(high)

        current = low

    approximation = current

    return details, approximation


# ============================================================
# GENERAL DWT
# ============================================================

def dwt_decompose(
    signal,
    wavelet,
    levels
):

    # Haar is implemented from scratch
    # so that our original DSP implementation
    # remains available.

    if wavelet == "haar":

        return haar_dwt(
            signal,
            levels
        )

    # Other wavelets use PyWavelets

    coeffs = pywt.wavedec(
        signal,
        wavelet,
        level=levels,
        mode="periodization"
    )

    # PyWavelets returns:
    #
    # [A3, D3, D2, D1]
    #
    # We want:
    #
    # [D1, D2, D3], A3

    approximation = coeffs[0]

    details = coeffs[
        1:
    ][::-1]

    return details, approximation


# ============================================================
# SWT
# ============================================================

def swt_decompose(
    signal,
    wavelet,
    levels
):

    coeffs = pywt.swt(
        signal,
        wavelet,
        level=levels,
        trim_approx=False,
        norm=False
    )

    # PyWavelets returns:
    #
    # [(A3,D3), (A2,D2), (A1,D1)]
    #
    # Convert to:
    #
    # details = [D1,D2,D3]
    # approximation = A3

    approximation = coeffs[0][0]

    details = [
        pair[1]
        for pair in coeffs[::-1]
    ]

    return details, approximation    


# ============================================================
# UNIVERSAL INTERFACE
# ============================================================

def decompose(
    signal,
    transform,
    wavelet,
    levels
):

    if wavelet not in SUPPORTED_WAVELETS:
        raise ValueError(
            f"Unsupported wavelet: {wavelet}. "
            f"Choose from {SUPPORTED_WAVELETS}"
        )

    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform: {transform}. "
            f"Choose from {SUPPORTED_TRANSFORMS}"
        )

    if transform == "dwt":

        return dwt_decompose(
            signal,
            wavelet,
            levels
        )

    else:

        return swt_decompose(
            signal,
            wavelet,
            levels
        )