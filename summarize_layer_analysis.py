"""Aggregate per-fold layer analysis CSV outputs into seed-level summaries."""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np


GROUPS = ('low', 'medium', 'high')


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fold_number(path):
    candidate = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if not candidate.startswith('fold') or not candidate[4:].isdigit():
        raise ValueError(f'Cannot determine fold number from {path}')
    return int(candidate[4:])


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def summarize_metric_rows(rows, key_fields, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)
    summaries = []
    for key, items in sorted(grouped.items()):
        summary = dict(zip(key_fields, key))
        summary['folds'] = len({int(item['fold']) for item in items})
        if 'count' in items[0]:
            counts = [numeric(item.get('count')) for item in items]
            counts = [value for value in counts if np.isfinite(value)]
            summary['mean_count'] = float(np.mean(counts)) if counts else np.nan
            summary['std_count'] = float(np.std(counts)) if counts else np.nan
        for metric in metric_fields:
            values = [numeric(item.get(metric)) for item in items]
            values = [value for value in values if np.isfinite(value)]
            summary[f'{metric}_mean'] = float(np.mean(values)) if values else np.nan
            summary[f'{metric}_std'] = float(np.std(values)) if values else np.nan
        summaries.append(summary)
    return summaries


def save_metric_plot(rows, item_field, metric_field, filename, title, ylabel, item_order):
    import matplotlib.pyplot as plt

    means = []
    stds = []
    per_item = defaultdict(list)
    for row in rows:
        value = numeric(row.get(metric_field))
        if np.isfinite(value):
            per_item[row[item_field]].append((int(row['fold']), value))
    count_by_item = defaultdict(list)
    for row in rows:
        count = numeric(row.get('count'))
        if np.isfinite(count):
            count_by_item[row[item_field]].append(count)
    for item in item_order:
        values = [value for _, value in sorted(per_item[item])]
        means.append(float(np.mean(values)) if values else np.nan)
        stds.append(float(np.std(values)) if values else np.nan)
    x = np.arange(len(item_order))
    fig, ax = plt.subplots(figsize=(max(8, len(item_order) * 1.1), 5))
    ax.errorbar(x, means, yerr=stds, fmt='o', capsize=4, label='fold mean ± std')
    for index, item in enumerate(item_order):
        points = sorted(per_item[item])
        if points:
            offsets = np.linspace(-0.12, 0.12, len(points)) if len(points) > 1 else np.array([0.0])
            ax.scatter(index + offsets, [value for _, value in points], s=30, label=None)
    ax.axhline(0, color='black', linewidth=0.8)
    tick_labels = [
        f'{item}\nn={np.mean(count_by_item[item]):.1f}' if count_by_item[item] else item
        for item in item_order
    ]
    ax.set_xticks(x, tick_labels, rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def require_complete_artifacts(seed_dir):
    expected = [os.path.join(seed_dir, f'fold{fold}') for fold in range(1, 6)]
    required_relative = [
        'cka/cka_matrix.csv', 'probe/probe_metrics.csv',
        'ablation/ablation_metrics.csv', 'ambiguity/probe_by_entropy.csv',
        'ambiguity/ablation_by_entropy.csv',
        'prediction_diagnostics/probe_by_ambiguity.csv',
        'prediction_diagnostics/prediction_distribution_deltas.csv',
        'prediction_diagnostics/prediction_entropy_by_ambiguity.csv',
        'prediction_diagnostics/class_probability_changes.csv',
        'prediction_diagnostics/entropy_group_boundaries.csv',
        'prediction_diagnostics/utterance_prediction_changes.csv',
        'prediction_diagnostics/focused_utterance_examples.csv',
    ]
    missing = [os.path.join(fold_dir, relpath)
               for fold_dir in expected for relpath in required_relative
               if not os.path.isfile(os.path.join(fold_dir, relpath))]
    if missing:
        raise FileNotFoundError('5-fold summary is incomplete; missing required files:\n' + '\n'.join(missing))


def aggregate_rows(paths, key_fields, metric_fields, output_path, eval_split='dev'):
    groups = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(list)
    for path in paths:
        for row in read_csv(path):
            if 'split' in row and row['split'] != eval_split:
                continue
            key = tuple(row[field] for field in key_fields)
            if 'count' in row:
                counts[key].append(float(row['count']))
            for metric in metric_fields:
                value = row.get(metric, '')
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    groups[key][metric].append(value)
    output_rows = []
    for key in sorted(set(groups) | set(counts)):
        row = dict(zip(key_fields, key))
        if counts[key]:
            row['mean_count'] = f'{np.mean(counts[key]):.4f}'
            row['std_count'] = f'{np.std(counts[key]):.4f}'
        for metric in metric_fields:
            values = groups[key][metric]
            row[f'{metric}_mean'] = f'{np.mean(values):.6f}' if values else 'NaN'
            row[f'{metric}_std'] = f'{np.std(values):.6f}' if values else 'NaN'
            row[f'{metric}_folds'] = len(values)
        output_rows.append(row)
    write_csv(output_path, output_rows)


def summarize(root, seed=1, eval_split='dev', require_all_folds=False):
    seed_dir = os.path.join(root, f'seed{seed}')
    fold_dirs = sorted(glob.glob(os.path.join(seed_dir, 'fold*')))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold analysis directories found under {seed_dir}')
    fold_numbers = sorted(int(os.path.basename(path)[4:]) for path in fold_dirs
                          if os.path.basename(path)[4:].isdigit())
    if require_all_folds and fold_numbers != [1, 2, 3, 4, 5]:
        raise ValueError(f'5-fold summary requires folds 1-5; found {fold_numbers}')
    if require_all_folds or fold_numbers == [1, 2, 3, 4, 5]:
        require_complete_artifacts(seed_dir)
    summary_dir = os.path.join(seed_dir, 'summary')
    figures_dir = os.path.join(summary_dir, 'figures')
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    cka_paths = [os.path.join(path, 'cka', 'cka_matrix.csv') for path in fold_dirs]
    cka_paths = [path for path in cka_paths if os.path.isfile(path)]
    if cka_paths:
        matrices = []
        labels = None
        cka_fold_rows = []
        for path in cka_paths:
            rows = read_csv(path)
            labels = [row['layer'] for row in rows]
            matrix = np.asarray([[float(row[name]) for name in labels] for row in rows])
            matrices.append(matrix)
            fold = fold_number(path)
            for label, values in zip(labels, matrix):
                cka_fold_rows.append({'fold': fold, 'layer': label,
                                      **{name: float(value) for name, value in zip(labels, values)}})
        write_csv(os.path.join(summary_dir, 'cka_fold_comparison.csv'), cka_fold_rows)
        matrix_stack = np.stack(matrices)
        for filename, values in (('cka_mean.csv', np.nanmean(matrix_stack, axis=0)),
                                 ('cka_std.csv', np.nanstd(matrix_stack, axis=0))):
            with open(os.path.join(summary_dir, filename), 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(['layer'] + labels)
                for label, row in zip(labels, values):
                    writer.writerow([label] + [f'{value:.8f}' for value in row])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(np.nanmean(matrix_stack, axis=0), vmin=0.0, vmax=1.0, cmap='viridis')
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title('Mean CKA across folds (Dev)')
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = np.nanmean(matrix_stack, axis=0)[row, column]
                ax.text(column, row, f'{value:.3f}' if np.isfinite(value) else 'NaN',
                        ha='center', va='center', color='white')
        fig.colorbar(image, ax=ax, label='Linear CKA')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, 'cka_5fold_mean_heatmap.png'), dpi=160)
        plt.close(fig)
        fused_paths = [os.path.join(path, 'cka', 'cka_fused_reference.csv') for path in fold_dirs]
        fused_paths = [path for path in fused_paths if os.path.isfile(path)]
        if fused_paths:
            fused_groups = defaultdict(list)
            for path in fused_paths:
                for row in read_csv(path):
                    try:
                        value = float(row['Fused_CKA'])
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        fused_groups[row['layer']].append(value)
            with open(os.path.join(summary_dir, 'cka_fused_reference_summary.csv'), 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(['layer', 'mean', 'std', 'folds'])
                for layer in [f'H{i}' for i in range(5)]:
                    values = fused_groups[layer]
                    writer.writerow([layer, f'{np.mean(values):.8f}' if values else 'NaN',
                                     f'{np.std(values):.8f}' if values else 'NaN', len(values)])

    probe_paths = glob.glob(os.path.join(seed_dir, 'fold*', 'probe', 'probe_metrics.csv'))
    if probe_paths:
        aggregate_rows(probe_paths, ['layer', 'split'], ['nll', 'kl', 'js', 'bc', 'tvd'],
                       os.path.join(summary_dir, 'probe_summary.csv'), eval_split)
        probe_fold_rows = []
        for path in probe_paths:
            for row in read_csv(path):
                if row.get('split') != eval_split:
                    continue
                probe_fold_rows.append({
                    'fold': fold_number(path), 'layer': row['layer'], 'split': row['split'],
                    **{name: numeric(row.get(name)) for name in ('nll', 'kl', 'tvd', 'js', 'bc')},
                    'count': numeric(row.get('count')),
                    'best_epoch': row.get('best_epoch', ''),
                })
        write_csv(os.path.join(summary_dir, 'probe_fold_comparison.csv'), probe_fold_rows)
        best_rows = []
        for fold in sorted({int(row['fold']) for row in probe_fold_rows}):
            candidates = [row for row in probe_fold_rows if int(row['fold']) == fold and np.isfinite(row['kl'])]
            if candidates:
                best = min(candidates, key=lambda row: row['kl'])
                best_rows.append({'fold': fold, 'best_layer_by_dev_kl': best['layer'],
                                  'dev_kl': best['kl'], 'dev_nll': best['nll'],
                                  'best_epoch': best['best_epoch']})
        write_csv(os.path.join(summary_dir, 'probe_best_layer_by_fold.csv'), best_rows)
        save_metric_plot(probe_fold_rows, 'layer', 'kl',
                         os.path.join(figures_dir, 'probe_kld_5fold.png'),
                         'Dev Probe KLD across folds', 'KLD (nats)',
                         [f'H{i}' for i in range(5)] + ['Fused'])

    ablation_paths = glob.glob(os.path.join(seed_dir, 'fold*', 'ablation', 'ablation_metrics.csv'))
    if ablation_paths:
        aggregate_rows(ablation_paths, ['split', 'condition'],
                       ['nll', 'delta_nll_vs_baseline', 'kl', 'js', 'bc', 'tvd'],
                       os.path.join(summary_dir, 'ablation_summary.csv'), eval_split)
        ablation_fold_rows = []
        for path in ablation_paths:
            rows = [row for row in read_csv(path) if row.get('split') == eval_split]
            baseline_rows = [row for row in rows if row.get('condition') == 'baseline']
            if not baseline_rows:
                continue
            baseline = baseline_rows[0]
            for row in rows:
                if row.get('condition') == 'baseline':
                    continue
                ablation_fold_rows.append({
                    'fold': fold_number(path), 'split': eval_split,
                    'condition': row['condition'], 'count': numeric(row.get('count')),
                    'delta_nll': numeric(row.get('nll')) - numeric(baseline.get('nll')),
                    'delta_kl': numeric(row.get('kl')) - numeric(baseline.get('kl')),
                    'delta_tvd': numeric(row.get('tvd')) - numeric(baseline.get('tvd')),
                })
        write_csv(os.path.join(summary_dir, 'ablation_fold_comparison.csv'), ablation_fold_rows)
        delta_summary = summarize_metric_rows(
            ablation_fold_rows, ['condition'], ['delta_nll', 'delta_kl', 'delta_tvd']
        )
        for row in delta_summary:
            matching = [item for item in ablation_fold_rows if item['condition'] == row['condition']]
            row['delta_kl_improved_folds'] = sum(numeric(item['delta_kl']) < 0 for item in matching)
            row['delta_kl_worsened_folds'] = sum(numeric(item['delta_kl']) > 0 for item in matching)
            row['delta_kl_sign_consistency'] = max(row['delta_kl_improved_folds'], row['delta_kl_worsened_folds'])
            row['delta_kl_sign_consistency_ratio'] = (
                row['delta_kl_sign_consistency'] / len(matching) if matching else np.nan
            )
        write_csv(os.path.join(summary_dir, 'ablation_delta_summary.csv'), delta_summary)
        save_metric_plot(ablation_fold_rows, 'condition', 'delta_kl',
                         os.path.join(figures_dir, 'ablation_delta_kl_5fold.png'),
                         'Dev Ablation ΔKLD across folds', 'ΔKLD (nats)',
                         [f'without_H{i}' for i in range(5)] + ['without_Fused'])

    ambiguity_probe_paths = glob.glob(os.path.join(seed_dir, 'fold*', 'ambiguity', 'probe_by_entropy.csv'))
    ambiguity_ablation_paths = glob.glob(os.path.join(seed_dir, 'fold*', 'ambiguity', 'ablation_by_entropy.csv'))
    ambiguity_outputs = []
    if ambiguity_probe_paths:
        aggregate_rows(ambiguity_probe_paths, ['split', 'entropy_group', 'layer'], ['mean_nll'],
                       os.path.join(summary_dir, 'probe_ambiguity_summary.csv'), eval_split)
        ambiguity_outputs.append(os.path.join(summary_dir, 'probe_ambiguity_summary.csv'))
    if ambiguity_ablation_paths:
        aggregate_rows(ambiguity_ablation_paths, ['split', 'entropy_group', 'condition'], ['mean_delta_nll'],
                       os.path.join(summary_dir, 'ablation_ambiguity_summary.csv'), eval_split)
        ambiguity_outputs.append(os.path.join(summary_dir, 'ablation_ambiguity_summary.csv'))
    if ambiguity_outputs:
        combined_rows = []
        for path in ambiguity_outputs:
            analysis_type = 'probe' if 'probe_' in os.path.basename(path) else 'ablation'
            for row in read_csv(path):
                item_field = 'layer' if analysis_type == 'probe' else 'condition'
                metric_field = 'mean_nll' if analysis_type == 'probe' else 'mean_delta_nll'
                combined_rows.append({
                    'analysis': analysis_type,
                    'split': row['split'],
                    'entropy_group': row['entropy_group'],
                    'item': row[item_field],
                    'mean_count': row.get('mean_count', ''),
                    'std_count': row.get('std_count', ''),
                    'metric': metric_field,
                    'metric_mean': row.get(f'{metric_field}_mean', 'NaN'),
                    'metric_std': row.get(f'{metric_field}_std', 'NaN'),
                    'folds': row.get(f'{metric_field}_folds', 0),
                })
        write_csv(os.path.join(summary_dir, 'ambiguity_summary.csv'), combined_rows)

    ambiguity_fold_rows = []
    diagnostic_probe_ambiguity_paths = glob.glob(os.path.join(
        seed_dir, 'fold*', 'prediction_diagnostics', 'probe_by_ambiguity.csv'))
    diagnostic_ablation_ambiguity_paths = glob.glob(os.path.join(
        seed_dir, 'fold*', 'prediction_diagnostics', 'prediction_entropy_by_ambiguity.csv'))
    for path in diagnostic_probe_ambiguity_paths:
        fold = fold_number(path)
        for row in read_csv(path):
            if row.get('split') != eval_split:
                continue
            ambiguity_fold_rows.append({
                'fold': fold, 'analysis': 'probe', 'split': eval_split,
                'entropy_group': row['entropy_group'], 'item': row['layer'],
                'count': numeric(row.get('count')), 'train_q33': numeric(row.get('train_q33')),
                'train_q67': numeric(row.get('train_q67')),
                'mean_nll': numeric(row.get('mean_nll')),
                'mean_kl': numeric(row.get('mean_kl')),
                'mean_tvd': numeric(row.get('mean_tvd')),
                'mean_delta_nll': np.nan, 'mean_delta_kl': np.nan, 'mean_delta_tvd': np.nan,
            })
    for path in diagnostic_ablation_ambiguity_paths:
        fold = fold_number(path)
        for row in read_csv(path):
            if row.get('split') != eval_split:
                continue
            if row.get('entropy_group') == 'all':
                continue
            ambiguity_fold_rows.append({
                'fold': fold, 'analysis': 'ablation', 'split': eval_split,
                'entropy_group': row['entropy_group'], 'item': row['condition'],
                'count': numeric(row.get('count')), 'train_q33': numeric(row.get('train_q33')),
                'train_q67': numeric(row.get('train_q67')),
                'mean_nll': np.nan, 'mean_kl': np.nan, 'mean_tvd': np.nan,
                'mean_delta_nll': numeric(row.get('mean_delta_nll')),
                'mean_delta_kl': numeric(row.get('mean_delta_kl')),
                'mean_delta_tvd': numeric(row.get('mean_delta_tvd')),
            })
    if ambiguity_fold_rows:
        write_csv(os.path.join(summary_dir, 'ambiguity_fold_comparison.csv'), ambiguity_fold_rows)
        ambiguity_summary_rows = []
        for analysis, metric_fields in (
            ('probe', ['mean_nll', 'mean_kl', 'mean_tvd']),
            ('ablation', ['mean_delta_nll', 'mean_delta_kl', 'mean_delta_tvd']),
        ):
            selected = [row for row in ambiguity_fold_rows if row['analysis'] == analysis]
            summaries = summarize_metric_rows(selected, ['entropy_group', 'item'], metric_fields)
            for summary in summaries:
                matched = [row for row in selected
                           if row['entropy_group'] == summary['entropy_group'] and row['item'] == summary['item']]
                summary['analysis'] = analysis
                summary['split'] = eval_split
                summary['mean_count'] = float(np.mean([r['count'] for r in matched])) if matched else np.nan
                summary['std_count'] = float(np.std([r['count'] for r in matched])) if matched else np.nan
                summary['train_q33_by_fold'] = ';'.join(f"{int(r['fold'])}:{r['train_q33']:.6g}" for r in matched)
                summary['train_q67_by_fold'] = ';'.join(f"{int(r['fold'])}:{r['train_q67']:.6g}" for r in matched)
                ambiguity_summary_rows.append(summary)
        write_csv(os.path.join(summary_dir, 'ambiguity_5fold_summary.csv'), ambiguity_summary_rows)

        for analysis, metric, filename, title, ylabel, items in (
            ('probe', 'mean_kl', 'probe_kld_by_ambiguity_5fold.png',
             'Dev Probe KLD by ambiguity across folds', 'KLD (nats)',
             [f'{layer}@{group}' for group in ('low', 'medium', 'high')
              for layer in [f'H{i}' for i in range(5)] + ['Fused']]),
            ('ablation', 'mean_delta_kl', 'ablation_delta_kl_by_ambiguity_5fold.png',
             'Dev Ablation ΔKLD by ambiguity across folds', 'ΔKLD (nats)',
             [f'without_{layer}@{group}' for group in ('low', 'medium', 'high')
              for layer in [f'H{i}' for i in range(5)] + ['Fused']]),
        ):
            rows = []
            for row in ambiguity_fold_rows:
                if row['analysis'] != analysis:
                    continue
                rows.append({**row, 'plot_item': f"{row['item']}@{row['entropy_group']}", 'plot_metric': row[metric]})
            save_metric_plot(rows, 'plot_item', 'plot_metric', os.path.join(figures_dir, filename),
                             title, ylabel, items)

    diagnostic_entropy_paths = glob.glob(os.path.join(
        seed_dir, 'fold*', 'prediction_diagnostics', 'prediction_entropy_by_ambiguity.csv'))
    diagnostic_class_paths = glob.glob(os.path.join(
        seed_dir, 'fold*', 'prediction_diagnostics', 'class_probability_changes.csv'))
    diagnostic_fold_rows = []
    entropy_metrics = [
        'mean_delta_kl', 'mean_delta_tvd', 'mean_delta_pred_entropy',
        'mean_delta_entropy_abs_error', 'improved_ratio', 'worsened_ratio',
        'max_abs_nll_entropy_kl_error', 'max_abs_ablation_nll_entropy_kl_error',
        'max_abs_delta_nll_kl_error',
    ]
    for path in diagnostic_entropy_paths:
        fold = fold_number(path)
        for row in read_csv(path):
            if row.get('split') != eval_split:
                continue
            diagnostic_fold_rows.append({
                'fold': fold, 'analysis': 'distribution', 'entropy_group': row['entropy_group'],
                'condition': row['condition'], 'class_name': '',
                'count': numeric(row.get('count')),
                'train_q33': numeric(row.get('train_q33')),
                'train_q67': numeric(row.get('train_q67')),
                **{metric: numeric(row.get(metric)) for metric in entropy_metrics},
                'mean_probability_change': np.nan,
                'baseline_mean_absolute_error': np.nan,
                'ablation_mean_absolute_error': np.nan,
            })
    for path in diagnostic_class_paths:
        fold = fold_number(path)
        for row in read_csv(path):
            if row.get('split') != eval_split:
                continue
            diagnostic_fold_rows.append({
                'fold': fold, 'analysis': 'class_probability',
                'entropy_group': row['entropy_group'], 'condition': row['condition'],
                'class_name': row['class_name'], 'count': numeric(row.get('count')),
                'train_q33': numeric(row.get('train_q33')),
                'train_q67': numeric(row.get('train_q67')),
                **{metric: np.nan for metric in entropy_metrics},
                'mean_probability_change': numeric(row.get('mean_probability_change')),
                'baseline_mean_absolute_error': numeric(row.get('baseline_mean_absolute_error')),
                'ablation_mean_absolute_error': numeric(row.get('ablation_mean_absolute_error')),
            })
    if diagnostic_fold_rows:
        write_csv(os.path.join(summary_dir, 'prediction_diagnostics_fold_comparison.csv'), diagnostic_fold_rows)
        diagnostic_summary = []
        diagnostic_key = ['analysis', 'entropy_group', 'condition', 'class_name']
        summary_metrics = entropy_metrics + [
            'mean_probability_change', 'baseline_mean_absolute_error', 'ablation_mean_absolute_error'
        ]
        # Aggregate per-fold means, never by concatenating repeated Dev utterances.
        for key in sorted({tuple(row[field] for field in diagnostic_key) for row in diagnostic_fold_rows}):
            matched = [row for row in diagnostic_fold_rows
                       if tuple(row[field] for field in diagnostic_key) == key]
            item = dict(zip(diagnostic_key, key))
            item['folds'] = len({int(row['fold']) for row in matched})
            counts = [row['count'] for row in matched if np.isfinite(row['count'])]
            item['mean_count'] = float(np.mean(counts)) if counts else np.nan
            item['std_count'] = float(np.std(counts)) if counts else np.nan
            item['train_q33_by_fold'] = ';'.join(
                f"{int(row['fold'])}:{row['train_q33']:.6g}" for row in matched
                if np.isfinite(row['train_q33']))
            item['train_q67_by_fold'] = ';'.join(
                f"{int(row['fold'])}:{row['train_q67']:.6g}" for row in matched
                if np.isfinite(row['train_q67']))
            for metric in summary_metrics:
                values = [row[metric] for row in matched if np.isfinite(row[metric])]
                item[f'{metric}_mean'] = float(np.mean(values)) if values else np.nan
                item[f'{metric}_std'] = float(np.std(values)) if values else np.nan
            improvements = [row['mean_delta_kl'] for row in matched
                            if row['analysis'] == 'distribution' and np.isfinite(row['mean_delta_kl'])]
            item['folds_delta_kl_negative'] = sum(value < 0 for value in improvements)
            item['folds_delta_kl_positive'] = sum(value > 0 for value in improvements)
            diagnostic_summary.append(item)
        write_csv(os.path.join(summary_dir, 'prediction_diagnostics_summary.csv'), diagnostic_summary)

        entropy_plot_rows = []
        for row in diagnostic_fold_rows:
            if row['analysis'] == 'distribution' and row['entropy_group'] in GROUPS:
                entropy_plot_rows.append({**row,
                    'plot_item': f"{row['condition']}@{row['entropy_group']}",
                    'plot_metric': row['mean_delta_pred_entropy']})
        save_metric_plot(entropy_plot_rows, 'plot_item', 'plot_metric',
                         os.path.join(figures_dir, 'prediction_entropy_change_5fold.png'),
                         'Prediction entropy change by ambiguity across folds',
                         'Δ predicted entropy',
                         [f'without_{layer}@{group}' for group in GROUPS
                          for layer in ('H2', 'H3')])

    with open(os.path.join(summary_dir, 'summary_info.txt'), 'w', encoding='utf-8') as output:
        output.write(f'Seed: {seed}\nEvaluation split: {eval_split}\n')
        output.write(f'Folds included: {len(fold_numbers)} ({", ".join(map(str, fold_numbers))})\n')
        output.write('Summary type: 5-fold\n' if fold_numbers == [1, 2, 3, 4, 5]
                     else f'Summary type: partial ({len(fold_numbers)} fold(s))\n')
    print(f'Saved fold summaries to {summary_dir} ({len(fold_numbers)} folds, {eval_split})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='layer_analysis')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval_split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--require_all_folds', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.seed, args.eval_split, args.require_all_folds)


if __name__ == '__main__':
    main()
