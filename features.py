import numpy as np


def scale_variance(coefficients):
    """
    Mean squared coefficient value for a wavelet scale.

    V_j = (1/N_j) * sum_k D_j[k]^2
    """
    coefficients = np.asarray(coefficients, dtype=float)

    if len(coefficients) == 0:
        return np.nan

    return np.mean(coefficients ** 2)


def energy(coefficients):
    """
    Raw wavelet energy:

    E_j = sum_k D_j[k]^2
    """
    coefficients = np.asarray(coefficients, dtype=float)

    return np.sum(coefficients ** 2)


def energy_shares(details, approximation):
    """
    Calculate additive energy shares across D1, D2, D3, A3.

    p_j = E_j / E_total
    """

    coefficients = list(details) + [approximation]

    energies = np.array(
        [energy(c) for c in coefficients],
        dtype=float
    )

    total_energy = np.sum(energies)

    if total_energy == 0:
        return np.full(len(energies), np.nan)

    return energies / total_energy


def wavelet_entropy(details, approximation):
    """
    Shannon entropy of the additive DWT energy distribution:

    H = -sum(p_j * ln(p_j))
    """

    shares = energy_shares(details, approximation)

    valid = shares > 0

    return -np.sum(
        shares[valid] * np.log(shares[valid])
    )


def high_low_ratio(details, approximation):
    """
    Ratio of short/intermediate-scale activity
    to long-scale activity.

    HF/LF = (V_D1 + V_D2) / (V_D3 + V_A3)
    """

    if len(details) < 3:
        raise ValueError(
            "high_low_ratio requires at least 3 detail levels."
        )

    v_d1 = scale_variance(details[0])
    v_d2 = scale_variance(details[1])
    v_d3 = scale_variance(details[2])
    v_a3 = scale_variance(approximation)

    low_frequency = v_d3 + v_a3

    if low_frequency == 0:
        return np.nan

    return (v_d1 + v_d2) / low_frequency


def extract_features(details, approximation):
    """
    Extract all wavelet features from one decomposition.

    details:
        [D1, D2, D3, ...]

    approximation:
        A3 (for a 3-level decomposition)
    """

    if len(details) < 3:
        raise ValueError(
            "At least 3 decomposition levels are required."
        )

    features = {
        "V_D1": scale_variance(details[0]),
        "V_D2": scale_variance(details[1]),
        "V_D3": scale_variance(details[2]),
        "V_A3": scale_variance(approximation),

        "HF_LF_ratio": high_low_ratio(
            details,
            approximation
        ),

        "wavelet_entropy": wavelet_entropy(
            details,
            approximation
        ),
    }

    shares = energy_shares(details, approximation)

    features["D1_energy_share"] = shares[0]
    features["D2_energy_share"] = shares[1]
    features["D3_energy_share"] = shares[2]
    features["A3_energy_share"] = shares[3]

    return features