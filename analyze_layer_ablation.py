"""Evaluate fixed-head sensitivity by replacing one feature block with its train mean."""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch

from dataloader import load_vocab
from distribution_metrics import calculate_distribution_metrics
from extract_layer_features import load_checkpoint
from model import DAGERC_multimodal


def load_features(fold_dir, split):
    with np.load(os.path.join(fold_dir, 'features', f'{split}_features.npz'), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def distribution_summary(targets, predictions):
    eps = 1e-12
    nll = -np.sum(targets * np.log(np.clip(predictions, eps, 1.0)), axis=1)
    rows = []
    for target, prediction in zip(targets, predictions):
        metrics = calculate_distribution_metrics(target, prediction)
        rows.append({
            'nll': float(-np.sum(target * np.log(np.clip(prediction, eps, 1.0)))),
            'kl': metrics['kl'], 'js': metrics['js'], 'bc': metrics['bc'],
            'tvd': float(0.5 * np.abs(target - prediction).sum()),
        })
    result = {'count': len(targets), 'nll': float(nll.mean()) if len(nll) else np.nan}
    for name in ('kl', 'js', 'bc', 'tvd'):
        values = [row[name] for row in rows if row[name] is not None and np.isfinite(row[name])]
        result[name] = float(np.mean(values)) if values else np.nan
    result['per_utterance_nll'] = nll
    return result


def analyze(checkpoint_path, fold_dir, output_dir=None, data_dir='output_data', device='auto'):
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint.get('args'), dict):
        raise ValueError('Checkpoint does not contain saved experiment arguments')
    args = SimpleNamespace(**checkpoint['args'])
    if getattr(args, 'nodal_att_type', None) is not None:
        raise ValueError('Layer ablation requires the Nodal Attention=None checkpoint')
    if getattr(args, 'raw_emotions', False) or not getattr(args, 'full', False):
        raise ValueError('Expected a full-mode checkpoint with aggregated five-class labels')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    _, vocab = load_vocab(data_dir, args)
    model = DAGERC_multimodal(args, len(vocab['itos']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    train = load_features(fold_dir, 'train')
    splits = {name: load_features(fold_dir, name) for name in ('train', 'dev', 'test')}
    layer_names = [f'H{i}' for i in range(args.gnn_layers + 1)] + ['Fused']
    means = {name: train[name].mean(axis=0) for name in layer_names}
    output_dir = output_dir or os.path.join(fold_dir, 'ablation')
    os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    utterance_rows = []
    for split_name, features in splits.items():
        with torch.inference_mode():
            pre_mlp = torch.as_tensor(features['pre_mlp'], dtype=torch.float32, device=device)
            baseline_predictions = torch.softmax(model.out_mlp(pre_mlp), dim=-1).cpu().numpy()
        stored_predictions = features['pred_prob']
        max_baseline_delta = float(np.max(np.abs(baseline_predictions - stored_predictions))) if len(stored_predictions) else 0.0
        if max_baseline_delta > 1e-4:
            raise ValueError(
                f'Checkpoint and extracted baseline predictions differ on {split_name}: '
                f'max abs diff={max_baseline_delta:.6g}'
            )

        condition_predictions = {'baseline': baseline_predictions}
        feature_offset = 0
        slices = {}
        for layer_name in layer_names:
            width = int(features[layer_name].shape[1])
            slices[layer_name] = slice(feature_offset, feature_offset + width)
            feature_offset += width
        if feature_offset != pre_mlp.shape[1]:
            raise ValueError(
                f'Concatenated feature width {feature_offset} does not match pre_mlp width {pre_mlp.shape[1]}'
            )

        for layer_name in layer_names:
            modified = pre_mlp.clone()
            modified[:, slices[layer_name]] = torch.as_tensor(
                means[layer_name], dtype=torch.float32, device=device
            )
            with torch.inference_mode():
                condition_predictions[f'without_{layer_name}'] = torch.softmax(
                    model.out_mlp(modified), dim=-1
                ).cpu().numpy()

        targets = features['true_soft']
        baseline_metrics = distribution_summary(targets, baseline_predictions)
        for condition, predictions in condition_predictions.items():
            metrics = distribution_summary(targets, predictions)
            metric_rows.append({
                'split': split_name, 'condition': condition, 'count': metrics['count'],
                'nll': metrics['nll'],
                'delta_nll_vs_baseline': metrics['nll'] - baseline_metrics['nll'],
                'kl': metrics['kl'], 'js': metrics['js'], 'bc': metrics['bc'],
                'tvd': metrics['tvd'],
                'baseline_prediction_max_abs_diff': max_baseline_delta,
            })
            if condition != 'baseline':
                deltas = metrics['per_utterance_nll'] - baseline_metrics['per_utterance_nll']
                for index, delta in enumerate(deltas):
                    utterance_rows.append({
                        'split': split_name,
                        'condition': condition,
                        'dialog_id': str(features['dialog_id'][index]),
                        'utterance_id': str(features['utterance_id'][index]),
                        'turn_index': int(features['turn_index'][index]),
                        'true_entropy': float(features['true_entropy'][index]),
                        'baseline_nll': float(baseline_metrics['per_utterance_nll'][index]),
                        'ablation_nll': float(metrics['per_utterance_nll'][index]),
                        'delta_nll': float(delta),
                    })
                np.savez_compressed(
                    os.path.join(output_dir, f'{split_name}_{condition}_predictions.npz'),
                    utterance_id=features['utterance_id'], dialog_id=features['dialog_id'],
                    true_soft=targets, pred_prob=predictions,
                    per_utterance_nll=metrics['per_utterance_nll'],
                )

    def write_csv(path, rows):
        with open(path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(os.path.join(output_dir, 'ablation_metrics.csv'), metric_rows)
    write_csv(os.path.join(output_dir, 'utterance_deltas.csv'), utterance_rows)

    import matplotlib.pyplot as plt
    test_rows = [row for row in metric_rows if row['split'] == 'test']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([row['condition'] for row in test_rows], [row['delta_nll_vs_baseline'] for row in test_rows])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('ΔNLL (without layer - baseline)')
    ax.set_title('Fixed-head layer ablation on test split')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'delta_nll.png'), dpi=160)
    plt.close(fig)
    print(f'Saved ablation results to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--data_dir', default='output_data')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    analyze(args.checkpoint, args.fold_dir, args.output_dir, args.data_dir, args.device)


if __name__ == '__main__':
    main()
