"""
Part 1: diversity combining experiment.

Students complete SC, MRC and a BER simulation over independent Rayleigh
flat fading branches.
"""

import numpy as np

from utils import (
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
    plot_diversity_snapshot,
    rayleigh_fading_branches,
)


def _validate_branch_arrays(received, channel):
    received = np.asarray(received, dtype=complex)
    channel = np.asarray(channel, dtype=complex)
    if received.ndim != 2 or channel.ndim != 2:
        raise ValueError('received and channel must be 2-D arrays: branches x symbols')
    if received.shape != channel.shape:
        raise ValueError('received and channel must have the same shape')
    if received.shape[0] < 1 or received.shape[1] < 1:
        raise ValueError('received and channel must not be empty')
    if np.any(np.abs(channel) < 1e-12):
        raise ValueError('channel contains near-zero coefficients')
    return received, channel


def selection_combining(received, channel):
    """
    Selection combining for flat fading branches.
    """
    received, channel = _validate_branch_arrays(received, channel)

    # Find the index of the branch with the maximum channel power for each symbol
    max_idx = np.argmax(np.abs(channel) ** 2, axis=0)
    # Create an array of symbol indices
    symbol_idx = np.arange(channel.shape[1])
    # Select the received signal and channel coefficient for the strongest branch
    best_r = received[max_idx, symbol_idx]
    best_h = channel[max_idx, symbol_idx]
    # Equalize by dividing the received signal by the channel coefficient
    return best_r / best_h


def maximal_ratio_combining(received, channel):
    """
    Maximal ratio combining for flat fading branches.
    """
    received, channel = _validate_branch_arrays(received, channel)

    # Multiply received signal by the complex conjugate of the channel
    numerator = np.sum(np.conj(channel) * received, axis=0)
    # Calculate the total channel power across all branches
    denominator = np.sum(np.abs(channel) ** 2, axis=0)
    # Return the normalized combined signal
    return numerator / denominator


def simulate_diversity_ber(snr_db_values, num_bits=4000, num_branches=2, seed=2026):
    """
    Simulate BER for no diversity, SC and MRC.
    """
    snr_db_values = np.asarray(snr_db_values, dtype=float)
    if snr_db_values.ndim != 1 or len(snr_db_values) == 0:
        raise ValueError('snr_db_values must be a non-empty one-dimensional array')
    if num_bits <= 0 or num_branches < 2:
        raise ValueError('num_bits must be positive and num_branches must be at least 2')

    # Initialize dictionary to store BER results
    ber_results = {'单分支': [], 'SC': [], 'MRC': []}

    # Generate random bits and modulate to BPSK symbols
    bits = generate_bits(num_bits, seed=seed)
    symbols = bpsk_modulate(bits)

    for snr in snr_db_values:
        # Simulate Rayleigh fading branches for the current SNR
        # Add SNR value to seed to ensure independent noise realizations across iterations
        current_seed = seed + int(snr) if seed is not None else None
        received, channel = rayleigh_fading_branches(symbols, num_branches, snr, current_seed)

        # 1. Single-branch equalization (using the first branch as baseline)
        single_rx = received[0] / channel[0]
        single_ber = calculate_ber(bits, bpsk_demodulate(single_rx))
        ber_results['单分支'].append(single_ber)

        # 2. Selection Combining
        sc_rx = selection_combining(received, channel)
        sc_ber = calculate_ber(bits, bpsk_demodulate(sc_rx))
        ber_results['SC'].append(sc_ber)

        # 3. Maximal Ratio Combining
        mrc_rx = maximal_ratio_combining(received, channel)
        mrc_ber = calculate_ber(bits, bpsk_demodulate(mrc_rx))
        ber_results['MRC'].append(mrc_ber)

    return ber_results


def equal_gain_combining(received, channel):
    """Optional: equal-gain combining with phase-only correction."""
    received, channel = _validate_branch_arrays(received, channel)

    # Perform phase-only correction using conjugate channel normalized by its magnitude
    phase_correction = np.conj(channel) / np.abs(channel)
    # Sum the phase-corrected received signals across all branches
    combined = np.sum(received * phase_correction, axis=0)
    # Equal gain combining does not require amplitude scaling for BPSK hard decision
    return combined


def run_diversity_demo():
    """Run Part 1 demo and generate figures."""
    print('=' * 60)
    print('Part 1: 分集合并实验')
    print('=' * 60)
    snr_db_values = np.array([0, 3, 6, 9, 12, 15], dtype=float)

    try:
        ber_curves = simulate_diversity_ber(snr_db_values, num_bits=6000, num_branches=2, seed=2026)
        plot_ber_curve(snr_db_values, ber_curves, '瑞利衰落信道下分集合并 BER 对比', 'diversity_ber_curve.png')

        bits = generate_bits(120, seed=7)
        symbols = bpsk_modulate(bits)
        received, channel = rayleigh_fading_branches(symbols, 2, snr_db=8, seed=17)
        branch_equalized = received[0] / channel[0]
        mrc_output = maximal_ratio_combining(received, channel)
        plot_diversity_snapshot(symbols, branch_equalized, mrc_output, 'diversity_waveform_snapshot.png')

        print('[OK] 已生成 results/diversity_ber_curve.png')
        print('[OK] 已生成 results/diversity_waveform_snapshot.png')
    except NotImplementedError as error:
        print(f'[WAIT] 尚未完成核心函数: {error}')
    except Exception as error:
        print(f'[FAIL] Part 1 运行失败: {error}')


if __name__ == '__main__':
    run_diversity_demo()
