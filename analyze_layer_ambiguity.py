"""Group probe and fixed-head ablation results by train-derived entropy tertiles."""
import argparse
import csv
import os

import numpy as np

from distribution_metrics import calculate_distribution_metrics


GROUPS = ('low', 'medium', 'high')


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        fields = ['split', 'entropy_group', 'count']
    else:
        fields = list(rows[0])
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_entropy(values, lower, upper):
    values = np.asarray(values)
    groups = np.full(values.shape, '', dtype=object)
    valid = np.isfinite(values)
    groups[valid & (values <= lower)] = 'low'
    groups[valid & (values > lower) & (values <= upper)] = 'medium'
    groups[valid & (values > upper)] = 'high'
    return groups


def keyed_order(reference, candidate, label):
    ref_keys = [(str(d), str(u)) for d, u in zip(reference['dialog_id'], reference['utterance_id'])]
    cand_keys = [(str(d), str(u)) for d, u in zip(candidate['dialog_id'], candidate['utterance_id'])]
    if len(set(ref_keys)) != len(ref_keys) or len(set(cand_keys)) != len(cand_keys):
        raise ValueError(f'Duplicate utterance IDs while aligning {label}')
    ref_index = {key: index for index, key in enumerate(ref_keys)}
    cand_index = {key: index for index, key in enumerate(cand_keys)}
    if ref_index.keys() != cand_index.keys():
        raise ValueError(f'{label} utterance IDs do not match the {reference.get("split", "eval")} features')
    return [cand_index[key] for key in ref_keys]


def kl_tvd(targets, predictions):
    kls, tvds = [], []
    for target, prediction in zip(targets, predictions):
        metrics = calculate_distribution_metrics(target, prediction)
        kls.append(metrics['kl'] if metrics['kl'] is not None else np.nan)
        tvds.append(0.5 * float(np.abs(target - prediction).sum()))
    return np.asarray(kls, dtype=np.float64), np.asarray(tvds, dtype=np.float64)


def analyze(fold_dir, output_dir=None, eval_split='dev'):
    feature_dir = os.path.join(fold_dir, 'features')
    train = load_npz(os.path.join(feature_dir, 'train_features.npz'))
    train_entropy = train['true_entropy'][np.isfinite(train['true_entropy'])]
    if len(train_entropy) == 0:
        raise ValueError('No valid train entropies are available to define ambiguity groups')
    lower, upper = np.quantile(train_entropy, [1 / 3, 2 / 3]).tolist()
    output_dir = output_dir or os.path.join(fold_dir, 'ambiguity')
    os.makedirs(output_dir, exist_ok=True)
    features = load_npz(os.path.join(feature_dir, f'{eval_split}_features.npz'))
    groups = group_entropy(features['true_entropy'], lower, upper)
    ranges = []
    for group in GROUPS:
        values = features['true_entropy'][groups == group]
        if not len(values):
            print(f'Warning: {eval_split} entropy group "{group}" is empty')
        ranges.append({
            'split': eval_split, 'entropy_group': group, 'count': int(len(values)),
            'train_q33': float(lower), 'train_q67': float(upper),
            'observed_entropy_min': float(np.min(values)) if len(values) else np.nan,
            'observed_entropy_max': float(np.max(values)) if len(values) else np.nan,
        })
    write_csv(os.path.join(output_dir, 'entropy_group_boundaries.csv'), ranges)

    probe_rows = []
    for layer in [f'H{i}' for i in range(5)] + ['Fused']:
        prediction_path = os.path.join(fold_dir, 'probe', 'predictions', f'{eval_split}_{layer}.npz')
        if not os.path.isfile(prediction_path):
            raise FileNotFoundError(f'Missing probe predictions: {prediction_path}')
        prediction = load_npz(prediction_path)
        prediction_order = keyed_order(features, prediction, f'Probe {layer}')
        pred_prob = prediction['pred_prob'][prediction_order]
        if not np.allclose(prediction['true_soft'][prediction_order], features['true_soft'], atol=1e-6):
            raise ValueError(f'Probe {layer} targets do not match {eval_split} features')
        probe_kl, probe_tvd = kl_tvd(features['true_soft'], pred_prob)
        for group in GROUPS:
            selected = groups == group
            ordered_values = prediction['per_utterance_nll'][prediction_order][selected]
            probe_rows.append({
                'split': eval_split, 'entropy_group': group, 'layer': layer,
                'count': int(selected.sum()),
                'mean_nll': float(np.mean(ordered_values)) if len(ordered_values) else np.nan,
                'mean_kl': float(np.mean(probe_kl[selected])) if selected.any() else np.nan,
                'mean_tvd': float(np.mean(probe_tvd[selected])) if selected.any() else np.nan,
                'train_q33': float(lower), 'train_q67': float(upper),
            })
    write_csv(os.path.join(output_dir, 'probe_by_entropy.csv'), probe_rows)

    ablation_rows_raw = read_csv(os.path.join(fold_dir, 'ablation', 'utterance_deltas.csv'))
    ablation_rows = []
    ids_to_group = {
        (str(features['dialog_id'][i]), str(features['utterance_id'][i])): groups[i]
        for i in range(len(groups))
    }
    split_rows = [row for row in ablation_rows_raw if row['split'] == eval_split]
    conditions = sorted({row['condition'] for row in split_rows})
    baseline_kl, baseline_tvd = kl_tvd(features['true_soft'], features['pred_prob'])
    for condition in conditions:
        prediction_path = os.path.join(fold_dir, 'ablation', f'{eval_split}_{condition}_predictions.npz')
        prediction = load_npz(prediction_path)
        prediction_order = keyed_order(features, prediction, f'Ablation {condition}')
        pred_prob = prediction['pred_prob'][prediction_order]
        if not np.allclose(prediction['true_soft'][prediction_order], features['true_soft'], atol=1e-6):
            raise ValueError(f'Ablation {condition} targets do not match {eval_split} features')
        ablation_kl, ablation_tvd = kl_tvd(features['true_soft'], pred_prob)
        condition_rows = [row for row in split_rows if row['condition'] == condition]
        delta_nll_by_key = {(row['dialog_id'], row['utterance_id']): float(row['delta_nll'])
                            for row in condition_rows}
        if delta_nll_by_key.keys() != ids_to_group.keys():
            raise ValueError(f'Ablation ΔNLL utterance IDs do not match {eval_split} features for {condition}')
        for group in GROUPS:
            selected = groups == group
            values = [delta_nll_by_key.get(key, np.nan) for key, assigned in ids_to_group.items()
                      if assigned == group]
            values = [value for value in values if np.isfinite(value)]
            ablation_rows.append({
                'split': eval_split, 'entropy_group': group, 'condition': condition,
                'count': int(selected.sum()),
                'mean_delta_nll': float(np.mean(values)) if values else np.nan,
                'mean_delta_kl': float(np.mean((ablation_kl - baseline_kl)[selected])) if selected.any() else np.nan,
                'mean_delta_tvd': float(np.mean((ablation_tvd - baseline_tvd)[selected])) if selected.any() else np.nan,
                'train_q33': float(lower), 'train_q67': float(upper),
            })
    write_csv(os.path.join(output_dir, 'ablation_by_entropy.csv'), ablation_rows)

    import matplotlib.pyplot as plt
    test_probe = probe_rows
    layers = [f'H{i}' for i in range(5)] + ['Fused']
    x = np.arange(len(GROUPS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, layer in enumerate(layers):
        values = [next(row['mean_nll'] for row in test_probe
                       if row['layer'] == layer and row['entropy_group'] == group)
                  for group in GROUPS]
        ax.bar(x + (index - (len(layers) - 1) / 2) * width, values, width, label=layer)
    ax.set_xticks(x, GROUPS)
    ax.set_ylabel(f'{eval_split.title()} NLL')
    ax.set_title('Layer probes by ground-truth ambiguity')
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'probe_by_entropy_{eval_split}.png'), dpi=160)
    plt.close(fig)

    test_ablation = ablation_rows
    conditions = [f'without_H{i}' for i in range(5)] + ['without_Fused']
    matrix = np.asarray([
        [next(row['mean_delta_nll'] for row in test_ablation
              if row['condition'] == condition and row['entropy_group'] == group)
         for group in GROUPS]
        for condition in conditions
    ], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(len(GROUPS)), GROUPS)
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_title(f'{eval_split.title()} ΔNLL by ambiguity group')
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text = f'{matrix[row, column]:.3f}' if np.isfinite(matrix[row, column]) else 'NaN'
            ax.text(column, row, text, ha='center', va='center')
    fig.colorbar(image, ax=ax, label='ΔNLL')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'delta_nll_heatmap_{eval_split}.png'), dpi=160)
    plt.close(fig)
    print(f'Entropy groups from train data: q33={lower:.6f}, q67={upper:.6f}')
    print(f'Saved ambiguity analyses to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--eval_split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    analyze(args.fold_dir, args.output_dir, args.eval_split)


if __name__ == '__main__':
    main()
