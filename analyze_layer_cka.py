"""Compute linear CKA between H0-H4 in one extracted feature file."""
import argparse
import csv
import os

import numpy as np


LAYERS = [f'H{i}' for i in range(5)]


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError('CKA inputs must be 2D matrices with the same sample count')
    if x.shape[0] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, 'fro') * np.linalg.norm(yy, 'fro')
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.sum(cross * cross) / denominator)


def analyze(input_path, output_dir=None):
    output_dir = output_dir or os.path.join(os.path.dirname(input_path), 'cka')
    os.makedirs(output_dir, exist_ok=True)
    with np.load(input_path, allow_pickle=True) as data:
        matrices = {name: data[name] for name in LAYERS}
        fused = data['Fused']
        utterance_ids = data['utterance_id']

    cka_matrix = np.full((len(LAYERS), len(LAYERS)), np.nan)
    for row, first in enumerate(LAYERS):
        for column, second in enumerate(LAYERS):
            cka_matrix[row, column] = linear_cka(matrices[first], matrices[second])

    csv_path = os.path.join(output_dir, 'cka_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['layer'] + LAYERS)
        for name, values in zip(LAYERS, cka_matrix):
            writer.writerow([name] + [f'{value:.8f}' if np.isfinite(value) else 'NaN' for value in values])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cka_matrix, vmin=0.0, vmax=1.0, cmap='viridis')
    ax.set_xticks(range(len(LAYERS)), LAYERS)
    ax.set_yticks(range(len(LAYERS)), LAYERS)
    ax.set_title(f'Linear CKA ({len(utterance_ids)} utterances)')
    for i in range(len(LAYERS)):
        for j in range(len(LAYERS)):
            label = f'{cka_matrix[i, j]:.3f}' if np.isfinite(cka_matrix[i, j]) else 'NaN'
            ax.text(j, i, label, ha='center', va='center', color='white')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_heatmap.png'), dpi=160)
    plt.close(fig)

    adjacent = np.array([cka_matrix[i, i + 1] for i in range(len(LAYERS) - 1)])
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f'{LAYERS[i]}-{LAYERS[i + 1]}' for i in range(len(LAYERS) - 1)]
    ax.plot(labels, adjacent, marker='o')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Linear CKA')
    ax.set_title('Adjacent-layer similarity')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_adjacent_layers.png'), dpi=160)
    plt.close(fig)
    fused_values = np.asarray([linear_cka(matrices[name], fused) for name in LAYERS])
    with open(os.path.join(output_dir, 'cka_fused_reference.csv'), 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['layer', 'Fused_CKA'])
        writer.writerows([
            [name, f'{value:.8f}' if np.isfinite(value) else 'NaN']
            for name, value in zip(LAYERS, fused_values)
        ])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(LAYERS, fused_values)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Linear CKA')
    ax.set_title('H0-H4 similarity to Fused (reference)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_fused_reference.png'), dpi=160)
    plt.close(fig)
    print(f'Saved CKA matrix and plots to {output_dir}')
    return cka_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='One split_features.npz file; dev is recommended')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    analyze(args.input, args.output_dir)


if __name__ == '__main__':
    main()
