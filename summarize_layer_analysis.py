"""Aggregate per-fold layer analysis CSV outputs into seed-level summaries."""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    summary_dir = os.path.join(seed_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    cka_paths = [os.path.join(path, 'cka', 'cka_matrix.csv') for path in fold_dirs]
    cka_paths = [path for path in cka_paths if os.path.isfile(path)]
    if cka_paths:
        matrices = []
        labels = None
        for path in cka_paths:
            rows = read_csv(path)
            labels = [row['layer'] for row in rows]
            matrices.append(np.asarray([[float(row[name]) for name in labels] for row in rows]))
        matrix_stack = np.stack(matrices)
        for filename, values in (('cka_mean.csv', np.nanmean(matrix_stack, axis=0)),
                                 ('cka_std.csv', np.nanstd(matrix_stack, axis=0))):
            with open(os.path.join(summary_dir, filename), 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(['layer'] + labels)
                for label, row in zip(labels, values):
                    writer.writerow([label] + [f'{value:.8f}' for value in row])
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

    ablation_paths = glob.glob(os.path.join(seed_dir, 'fold*', 'ablation', 'ablation_metrics.csv'))
    if ablation_paths:
        aggregate_rows(ablation_paths, ['split', 'condition'],
                       ['nll', 'delta_nll_vs_baseline', 'kl', 'js', 'bc', 'tvd'],
                       os.path.join(summary_dir, 'ablation_summary.csv'), eval_split)

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
