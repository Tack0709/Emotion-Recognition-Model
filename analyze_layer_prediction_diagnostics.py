"""Inspect how saved layer-ablation predictions change distribution metrics."""
import argparse
import csv
import json
import os

import numpy as np

from distribution_metrics import calculate_distribution_metrics


LAYERS = [f'H{i}' for i in range(5)] + ['Fused']
GROUPS = ('low', 'medium', 'high')
EPS = 1e-12


def load_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = probabilities.sum(axis=-1, keepdims=True)
    normalized = probabilities / np.maximum(total, EPS)
    logs = np.log(np.maximum(normalized, EPS))
    return -np.sum(np.where(normalized > 0, normalized * logs, 0.0), axis=-1)


def keyed_indices(data, label):
    required = ('dialog_id', 'utterance_id')
    if any(key not in data for key in required):
        raise ValueError(f'{label} does not contain dialog_id and utterance_id')
    keys = [(str(dialog), str(utterance))
            for dialog, utterance in zip(data['dialog_id'], data['utterance_id'])]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{label} contains duplicate (dialog_id, utterance_id) keys')
    return {key: index for index, key in enumerate(keys)}


def aligned_predictions(baseline, ablation, label):
    baseline_keys = keyed_indices(baseline, 'baseline features')
    ablation_keys = keyed_indices(ablation, label)
    if baseline_keys.keys() != ablation_keys.keys():
        missing = len(baseline_keys.keys() - ablation_keys.keys())
        extra = len(ablation_keys.keys() - baseline_keys.keys())
        raise ValueError(f'{label} utterance IDs do not match baseline (missing={missing}, extra={extra})')
    order = [ablation_keys[key] for key in baseline_keys]
    target = np.asarray(ablation['true_soft'], dtype=np.float64)[order]
    baseline_target = np.asarray(baseline['true_soft'], dtype=np.float64)
    if target.shape != baseline_target.shape or not np.allclose(target, baseline_target, atol=1e-6, rtol=1e-6):
        raise ValueError(f'{label} true_soft values do not match baseline by utterance ID')
    prediction = np.asarray(ablation['pred_prob'], dtype=np.float64)[order]
    return prediction


def entropy_groups(train_entropy, eval_entropy):
    values = np.asarray(train_entropy, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Train features contain no finite true_entropy values')
    q33, q67 = np.quantile(values, [1 / 3, 2 / 3]).tolist()
    eval_entropy = np.asarray(eval_entropy, dtype=np.float64)
    groups = np.full(len(eval_entropy), 'unknown', dtype=object)
    valid = np.isfinite(eval_entropy)
    groups[valid & (eval_entropy <= q33)] = 'low'
    groups[valid & (eval_entropy > q33) & (eval_entropy <= q67)] = 'medium'
    groups[valid & (eval_entropy > q67)] = 'high'
    return q33, q67, groups


def distribution_metrics(targets, predictions):
    output = {name: [] for name in ('kl', 'tvd', 'nll', 'entropy', 'entropy_abs_error')}
    for target, prediction in zip(targets, predictions):
        metrics = calculate_distribution_metrics(target, prediction)
        pred_entropy = float(entropy(prediction))
        true_entropy = float(entropy(target))
        output['kl'].append(metrics['kl'])
        output['tvd'].append(0.5 * float(np.abs(target - prediction).sum()))
        output['nll'].append(float(-np.sum(target * np.log(np.maximum(prediction, EPS)))))
        output['entropy'].append(pred_entropy)
        output['entropy_abs_error'].append(abs(pred_entropy - true_entropy))
    return {key: np.asarray(values, dtype=np.float64) for key, values in output.items()}


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def analyze(fold_dir, output_dir=None, eval_split='dev', top_k=20):
    if eval_split != 'dev':
        raise ValueError('Prediction diagnostics are exploratory and currently restricted to Dev data')
    if top_k < 0:
        raise ValueError('top_k must be nonnegative')
    feature_dir = os.path.join(fold_dir, 'features')
    baseline = load_npz(os.path.join(feature_dir, f'{eval_split}_features.npz'))
    train = load_npz(os.path.join(feature_dir, 'train_features.npz'))
    targets = np.asarray(baseline['true_soft'], dtype=np.float64)
    baseline_predictions = np.asarray(baseline['pred_prob'], dtype=np.float64)
    if targets.shape != baseline_predictions.shape or targets.ndim != 2:
        raise ValueError('Baseline true_soft and pred_prob must have matching [utterance, class] shapes')
    if not np.isfinite(targets).all() or not np.isfinite(baseline_predictions).all():
        raise ValueError('Baseline distributions contain non-finite values')
    if np.any(targets < 0) or np.any(baseline_predictions < 0):
        raise ValueError('Distributions must be nonnegative')
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError('Some true_soft rows do not sum to one')
    if not np.allclose(baseline_predictions.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError('Some baseline predictions do not sum to one')

    class_names = [str(name) for name in baseline.get('class_names', np.arange(targets.shape[1]))]
    if len(class_names) != targets.shape[1]:
        raise ValueError('class_names length does not match the probability dimension')
    baseline_keys = keyed_indices(baseline, 'baseline features')
    q33, q67, groups = entropy_groups(train['true_entropy'], baseline['true_entropy'])
    baseline_metrics = distribution_metrics(targets, baseline_predictions)
    true_entropy = entropy(targets)
    if 'true_entropy' in baseline and not np.allclose(
        np.asarray(baseline['true_entropy'], dtype=np.float64), true_entropy, atol=1e-5, rtol=1e-5
    ):
        raise ValueError('Stored true_entropy does not match entropy recomputed from true_soft')
    entropy_limit = float(np.log(targets.shape[1]))
    for label, values in [('true', true_entropy), ('baseline prediction', baseline_metrics['entropy'])]:
        if np.any(values < -1e-8) or np.any(values > entropy_limit + 1e-6):
            raise ValueError(f'{label} entropy is outside [0, ln(C)]')
    fold = int(np.asarray(baseline.get('fold', 0)).item())
    output_dir = output_dir or os.path.join(fold_dir, 'prediction_diagnostics')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    delta_rows = []
    entropy_rows = []
    class_rows = []
    utterance_rows = []
    probe_group_rows = []
    probe_dir = os.path.join(fold_dir, 'probe', 'predictions')
    for layer in LAYERS:
        probe = load_npz(os.path.join(probe_dir, f'{eval_split}_{layer}.npz'))
        probe_order = keyed_indices(baseline, 'baseline features')
        probe_index = keyed_indices(probe, f'Probe {layer}')
        if probe_index.keys() != probe_order.keys():
            raise ValueError(f'Probe {layer} utterance IDs do not match baseline')
        order = [probe_index[key] for key in probe_order]
        probe_targets = np.asarray(probe['true_soft'], dtype=np.float64)[order]
        probe_predictions = np.asarray(probe['pred_prob'], dtype=np.float64)[order]
        if not np.allclose(probe_targets, targets, atol=1e-6, rtol=1e-6):
            raise ValueError(f'Probe {layer} targets do not match baseline by utterance ID')
        if (probe_predictions.shape != targets.shape or not np.isfinite(probe_predictions).all()
                or np.any(probe_predictions < 0)
                or not np.allclose(probe_predictions.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError(f'Probe {layer} contains invalid probability distributions')
        probe_metrics = distribution_metrics(probe_targets, probe_predictions)
        for group in GROUPS + ('all',):
            selected = np.ones(len(groups), dtype=bool) if group == 'all' else groups == group
            probe_group_rows.append({
                'fold': fold, 'split': eval_split, 'entropy_group': group,
                'layer': layer, 'count': int(selected.sum()), 'train_q33': float(q33),
                'train_q67': float(q67),
                'mean_nll': finite_mean(probe_metrics['nll'][selected]),
                'mean_kl': finite_mean(probe_metrics['kl'][selected]),
                'mean_tvd': finite_mean(probe_metrics['tvd'][selected]),
            })
    for layer in LAYERS:
        condition = f'without_{layer}'
        ablation_path = os.path.join(fold_dir, 'ablation', f'{eval_split}_{condition}_predictions.npz')
        ablation = load_npz(ablation_path)
        predictions = aligned_predictions(baseline, ablation, condition)
        if predictions.shape != targets.shape:
            raise ValueError(f'{condition} prediction shape does not match baseline targets')
        if not np.isfinite(predictions).all() or np.any(predictions < 0):
            raise ValueError(f'{condition} contains invalid probabilities')
        probability_sums = predictions.sum(axis=1)
        if not np.allclose(probability_sums, 1.0, atol=1e-4):
            raise ValueError(f'{condition} prediction rows do not sum to one')

        metrics = distribution_metrics(targets, predictions)
        if np.any(metrics['entropy'] < -1e-8) or np.any(metrics['entropy'] > entropy_limit + 1e-6):
            raise ValueError(f'{condition} entropy is outside [0, ln(C)]')
        delta_kl = metrics['kl'] - baseline_metrics['kl']
        delta_tvd = metrics['tvd'] - baseline_metrics['tvd']
        delta_entropy = metrics['entropy'] - baseline_metrics['entropy']
        delta_entropy_abs_error = metrics['entropy_abs_error'] - baseline_metrics['entropy_abs_error']
        nll_kl_identity_error = np.abs(baseline_metrics['nll'] - true_entropy - baseline_metrics['kl'])
        ablation_nll_kl_identity_error = np.abs(metrics['nll'] - true_entropy - metrics['kl'])
        delta_nll_kl_error = np.abs((metrics['nll'] - baseline_metrics['nll']) - delta_kl)
        fold_value = int(np.asarray(baseline.get('fold', fold)).item())
        keys = list(baseline_keys)
        for index, key in enumerate(keys):
            base_row = {
                'fold': fold_value, 'split': eval_split, 'dialog_id': key[0],
                'utterance_id': key[1], 'turn_index': int(baseline['turn_index'][index]),
                'entropy_group': str(groups[index]), 'condition': condition,
                'true_entropy': float(true_entropy[index]),
                'baseline_pred_entropy': float(baseline_metrics['entropy'][index]),
                'ablation_pred_entropy': float(metrics['entropy'][index]),
                'delta_pred_entropy': float(delta_entropy[index]),
                'baseline_entropy_abs_error': float(baseline_metrics['entropy_abs_error'][index]),
                'ablation_entropy_abs_error': float(metrics['entropy_abs_error'][index]),
                'delta_entropy_abs_error': float(delta_entropy_abs_error[index]),
                'baseline_nll': float(baseline_metrics['nll'][index]),
                'ablation_nll': float(metrics['nll'][index]),
                'delta_nll': float(metrics['nll'][index] - baseline_metrics['nll'][index]),
                'baseline_kl': float(baseline_metrics['kl'][index]),
                'ablation_kl': float(metrics['kl'][index]),
                'delta_kl': float(delta_kl[index]),
                'baseline_tvd': float(baseline_metrics['tvd'][index]),
                'ablation_tvd': float(metrics['tvd'][index]),
                'delta_tvd': float(delta_tvd[index]),
                'true_soft': json.dumps(targets[index].tolist()),
                'baseline_pred_prob': json.dumps(baseline_predictions[index].tolist()),
                'ablation_pred_prob': json.dumps(predictions[index].tolist()),
                'text': str(baseline.get('text', np.full(len(targets), ''))[index]),
            }
            for class_index, class_name in enumerate(class_names):
                base_row[f'true_{class_name}'] = float(targets[index, class_index])
                base_row[f'baseline_{class_name}'] = float(baseline_predictions[index, class_index])
                base_row[f'ablation_{class_name}'] = float(predictions[index, class_index])
                base_row[f'delta_{class_name}'] = float(predictions[index, class_index] - baseline_predictions[index, class_index])
                base_row[f'baseline_error_{class_name}'] = float(baseline_predictions[index, class_index] - targets[index, class_index])
                base_row[f'ablation_error_{class_name}'] = float(predictions[index, class_index] - targets[index, class_index])
            delta_rows.append(base_row)
            if str(groups[index]) in GROUPS:
                utterance_rows.append({**base_row,
                                       'focus_priority': bool(groups[index] == 'high' and layer in ('H2', 'H3')),
                                       'change_direction':
                                       'improved' if delta_kl[index] < 0 else
                                       'worsened' if delta_kl[index] > 0 else 'unchanged'})

        for group in GROUPS + ('all',):
            selected = np.ones(len(groups), dtype=bool) if group == 'all' else groups == group
            entropy_rows.append({
                'fold': fold_value, 'split': eval_split, 'entropy_group': group,
                'condition': condition,
                'count': int(selected.sum()), 'train_q33': float(q33), 'train_q67': float(q67),
                'mean_true_entropy': finite_mean(true_entropy[selected]),
                'mean_baseline_pred_entropy': finite_mean(baseline_metrics['entropy'][selected]),
                'mean_ablation_pred_entropy': finite_mean(metrics['entropy'][selected]),
                'mean_delta_pred_entropy': finite_mean(delta_entropy[selected]),
                'mean_baseline_entropy_abs_error': finite_mean(baseline_metrics['entropy_abs_error'][selected]),
                'mean_ablation_entropy_abs_error': finite_mean(metrics['entropy_abs_error'][selected]),
                'mean_delta_entropy_abs_error': finite_mean(delta_entropy_abs_error[selected]),
                'mean_baseline_nll': finite_mean(baseline_metrics['nll'][selected]),
                'mean_ablation_nll': finite_mean(metrics['nll'][selected]),
                'mean_delta_nll': finite_mean((metrics['nll'] - baseline_metrics['nll'])[selected]),
                'max_abs_nll_entropy_kl_error': float(np.max(nll_kl_identity_error[selected])) if selected.any() else np.nan,
                'max_abs_ablation_nll_entropy_kl_error': float(np.max(ablation_nll_kl_identity_error[selected])) if selected.any() else np.nan,
                'max_abs_delta_nll_kl_error': float(np.max(delta_nll_kl_error[selected])) if selected.any() else np.nan,
                'mean_baseline_kl': finite_mean(baseline_metrics['kl'][selected]),
                'mean_ablation_kl': finite_mean(metrics['kl'][selected]),
                'mean_delta_kl': finite_mean(delta_kl[selected]),
                'mean_baseline_tvd': finite_mean(baseline_metrics['tvd'][selected]),
                'mean_ablation_tvd': finite_mean(metrics['tvd'][selected]),
                'mean_delta_tvd': finite_mean(delta_tvd[selected]),
                'improved_count': int(np.sum(delta_kl[selected] < 0)),
                'worsened_count': int(np.sum(delta_kl[selected] > 0)),
                'improved_ratio': finite_mean(delta_kl[selected] < 0),
                'worsened_ratio': finite_mean(delta_kl[selected] > 0),
            })
            for class_index, class_name in enumerate(class_names):
                t = targets[selected, class_index]
                b = baseline_predictions[selected, class_index]
                a = predictions[selected, class_index]
                class_rows.append({
                    'fold': fold_value, 'split': eval_split, 'entropy_group': group,
                    'condition': condition, 'class_name': class_name,
                    'train_q33': float(q33), 'train_q67': float(q67),
                    'count': int(selected.sum()), 'mean_true_probability': finite_mean(t),
                    'mean_baseline_probability': finite_mean(b),
                    'mean_ablation_probability': finite_mean(a),
                    'mean_probability_change': finite_mean(a - b),
                    'baseline_mean_absolute_error': finite_mean(np.abs(b - t)),
                    'ablation_mean_absolute_error': finite_mean(np.abs(a - t)),
                })

    write_csv(os.path.join(output_dir, 'prediction_distribution_deltas.csv'), delta_rows)
    write_csv(os.path.join(output_dir, 'prediction_entropy_by_ambiguity.csv'), entropy_rows)
    write_csv(os.path.join(output_dir, 'class_probability_changes.csv'), class_rows)
    write_csv(os.path.join(output_dir, 'probe_by_ambiguity.csv'), probe_group_rows)
    utterance_rows.sort(key=lambda row: (not row['focus_priority'], float(row['delta_kl'])))
    write_csv(os.path.join(output_dir, 'utterance_prediction_changes.csv'), utterance_rows)
    focused_examples = []
    for condition in ('without_H2', 'without_H3'):
        selected = [row for row in utterance_rows
                    if row['condition'] == condition and row['entropy_group'] == 'high']
        improved = sorted((row for row in selected if row['delta_kl'] < 0),
                          key=lambda row: float(row['delta_kl']))[:top_k]
        worsened = sorted((row for row in selected if row['delta_kl'] > 0),
                          key=lambda row: float(row['delta_kl']), reverse=True)[:top_k]
        focused_examples.extend(improved)
        focused_examples.extend(worsened)
    write_csv(os.path.join(output_dir, 'focused_utterance_examples.csv'), focused_examples)
    write_csv(os.path.join(output_dir, 'entropy_group_boundaries.csv'), [
        {'fold': fold, 'split': eval_split, 'train_q33': float(q33), 'train_q67': float(q67),
         'entropy_group': group, 'count': int(np.sum(groups == group)),
         'observed_entropy_min': finite_mean(np.min(baseline['true_entropy'][groups == group])) if np.any(groups == group) else np.nan,
         'observed_entropy_max': finite_mean(np.max(baseline['true_entropy'][groups == group])) if np.any(groups == group) else np.nan}
        for group in GROUPS
    ])

    # Check the NLL = entropy + KL identity using the exact saved distributions.
    relation_deltas = [abs(row['baseline_nll'] - row['true_entropy'] - row['baseline_kl'])
                       for row in delta_rows]
    relation_deltas.extend(abs(row['ablation_nll'] - row['true_entropy'] - row['ablation_kl'])
                           for row in delta_rows)
    if relation_deltas and max(relation_deltas) > 1e-5:
        print(f'Warning: max |NLL - H(y) - KL| = {max(relation_deltas):.3g}')
    nll_kl_deltas = [abs(row['delta_nll'] - row['delta_kl']) for row in delta_rows]
    if nll_kl_deltas and max(nll_kl_deltas) > 1e-5:
        print(f'Warning: max |ΔNLL - ΔKL| = {max(nll_kl_deltas):.3g}')

    make_figures(figures_dir, delta_rows, entropy_rows, class_rows, class_names)
    print(f'Saved prediction diagnostics to {output_dir}')


def make_figures(figures_dir, delta_rows, entropy_rows, class_rows, class_names):
    import matplotlib.pyplot as plt

    for metric, filename, ylabel in (
        ('delta_kl', 'delta_kl_vs_delta_entropy.png', 'ΔKLD (ablation - baseline)'),
        ('delta_tvd', 'delta_tvd_vs_delta_entropy.png', 'ΔTVD (ablation - baseline)'),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for group in GROUPS:
            rows = [row for row in delta_rows if row['entropy_group'] == group]
            ax.scatter([row['delta_pred_entropy'] for row in rows],
                       [row[metric] for row in rows], s=12, alpha=0.45, label=group)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Δ predicted entropy (ablation - baseline)')
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, filename), dpi=160)
        plt.close(fig)

    conditions = [f'without_{layer}' for layer in LAYERS]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(GROUPS))
    width = 0.12
    for index, condition in enumerate(conditions):
        rows = [row for row in entropy_rows if row['condition'] == condition and row['entropy_group'] in GROUPS]
        values = [next((row['mean_delta_pred_entropy'] for row in rows if row['entropy_group'] == group), np.nan)
                  for group in GROUPS]
        ax.bar(x + (index - (len(conditions) - 1) / 2) * width, values, width, label=condition)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x, GROUPS)
    ax.set_ylabel('Δ predicted entropy')
    ax.set_title('Prediction entropy change by ambiguity group')
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'prediction_entropy_by_ambiguity.png'), dpi=160)
    plt.close(fig)

    selected_conditions = ['without_H2', 'without_H3']
    fig, axes = plt.subplots(1, len(selected_conditions), figsize=(6 * len(selected_conditions), 5), squeeze=False)
    for ax, condition in zip(axes[0], selected_conditions):
        rows = [row for row in class_rows if row['condition'] == condition and row['entropy_group'] in GROUPS]
        width = 0.24
        x = np.arange(len(class_names))
        for j, group in enumerate(GROUPS):
            values = [next((row['mean_probability_change'] for row in rows
                            if row['entropy_group'] == group and row['class_name'] == name), np.nan)
                      for name in class_names]
            ax.bar(x + (j - 1) * width, values, width, label=group)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x, class_names, rotation=30)
        ax.set_title(condition)
        ax.set_ylabel('Mean probability change')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'class_probability_changes_by_ambiguity.png'), dpi=160)
    plt.close(fig)

    mass_rows = [row for row in class_rows if row['entropy_group'] == 'all']
    selected_conditions = ['without_H2', 'without_H3']
    fig, axes = plt.subplots(1, len(selected_conditions), figsize=(7 * len(selected_conditions), 5), squeeze=False)
    for ax, condition in zip(axes[0], selected_conditions):
        rows = [row for row in mass_rows if row['condition'] == condition]
        x = np.arange(len(class_names))
        width = 0.24
        for j, (field, label) in enumerate((('mean_true_probability', 'True'),
                                            ('mean_baseline_probability', 'Baseline'),
                                            ('mean_ablation_probability', 'Ablation'))):
            values = [next((row[field] for row in rows if row['class_name'] == name), np.nan)
                      for name in class_names]
            ax.bar(x + (j - 1) * width, values, width, label=label)
        ax.set_xticks(x, class_names, rotation=30)
        ax.set_ylabel('Mean class probability')
        ax.set_title(condition)
        ax.legend()
    fig.suptitle('True and predicted class probability mass')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'prediction_mass_comparison.png'), dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--eval_split', choices=['dev'], default='dev')
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()
    analyze(args.fold_dir, args.output_dir, args.eval_split, args.top_k)


if __name__ == '__main__':
    main()
