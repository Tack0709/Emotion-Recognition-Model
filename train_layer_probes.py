"""Train frozen-feature linear probes for H0-H4 and the Fused reference."""
import argparse
import csv
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distribution_metrics import calculate_distribution_metrics


LAYERS = [f'H{i}' for i in range(5)] + ['Fused']


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def predict(head, features, device, batch_size):
    head.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = head(torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device))
            predictions.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty((0, 0), dtype=np.float32)


def evaluate(targets, predictions):
    eps = 1e-12
    nll = -np.sum(targets * np.log(np.clip(predictions, eps, 1.0)), axis=1)
    kl, js, bc, tvd = [], [], [], []
    for target, prediction in zip(targets, predictions):
        metrics = calculate_distribution_metrics(target, prediction)
        if metrics['kl'] is not None:
            kl.append(metrics['kl'])
        if metrics['js'] is not None:
            js.append(metrics['js'])
        if metrics['bc'] is not None:
            bc.append(metrics['bc'])
        tvd.append(0.5 * float(np.abs(target - prediction).sum()))
    return {
        'nll': float(nll.mean()) if len(nll) else np.nan,
        'kl': float(np.mean(kl)) if kl else np.nan,
        'js': float(np.mean(js)) if js else np.nan,
        'bc': float(np.mean(bc)) if bc else np.nan,
        'tvd': float(np.mean(tvd)) if tvd else np.nan,
        'per_utterance_nll': nll,
    }


def train_one_layer(layer, train, dev, eval_data, eval_split, output_dir, device, seed, epochs, batch_size, learning_rate):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    head = nn.Linear(train[layer].shape[1], train['true_soft'].shape[1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    x_train = torch.as_tensor(train[layer], dtype=torch.float32)
    y_train = torch.as_tensor(train['true_soft'], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                        shuffle=True, generator=generator)

    best_dev_nll = float('inf')
    best_epoch = 0
    best_state = None
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        head.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            log_probs = torch.log_softmax(head(features), dim=-1)
            loss = -(targets * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_predictions = predict(head, dev[layer], device, batch_size)
        dev_metrics = evaluate(dev['true_soft'], dev_predictions)
        epoch_rows.append((epoch, dev_metrics['nll']))
        if np.isfinite(dev_metrics['nll']) and dev_metrics['nll'] < best_dev_nll:
            best_dev_nll = dev_metrics['nll']
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f'No finite dev NLL was obtained for {layer}')
    head.load_state_dict(best_state)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    torch.save({'state_dict': best_state, 'best_epoch': best_epoch, 'layer': layer,
                'seed': seed, 'input_dim': int(train[layer].shape[1]),
                'num_classes': int(train['true_soft'].shape[1])},
               os.path.join(output_dir, f'{layer}_probe.pt'))

    rows = []
    eval_sets = [('train', train)]
    if eval_split != 'train':
        eval_sets.append((eval_split, eval_data))
    for split_name, split_data in eval_sets:
        probabilities = predict(head, split_data[layer], device, batch_size)
        metrics = evaluate(split_data['true_soft'], probabilities)
        rows.append({
            'layer': layer, 'split': split_name, 'count': len(probabilities),
            'input_dim': int(train[layer].shape[1]),
            'best_epoch': best_epoch, 'nll': metrics['nll'], 'kl': metrics['kl'],
            'js': metrics['js'], 'bc': metrics['bc'], 'tvd': metrics['tvd'],
        })
        np.savez_compressed(
            os.path.join(output_dir, 'predictions', f'{split_name}_{layer}.npz'),
            utterance_id=split_data['utterance_id'], dialog_id=split_data['dialog_id'],
            turn_index=split_data['turn_index'], true_entropy=split_data['true_entropy'],
            true_soft=split_data['true_soft'], pred_prob=probabilities,
            per_utterance_nll=metrics['per_utterance_nll'],
        )
    return rows, epoch_rows


def train_probes(fold_dir, output_dir=None, seed=1, epochs=100, batch_size=256,
                 learning_rate=1e-3, device='auto', eval_split='dev'):
    feature_dir = os.path.join(fold_dir, 'features')
    train = load_npz(os.path.join(feature_dir, 'train_features.npz'))
    dev = load_npz(os.path.join(feature_dir, 'dev_features.npz'))
    eval_data = dev if eval_split == 'dev' else load_npz(os.path.join(feature_dir, 'test_features.npz'))
    output_dir = output_dir or os.path.join(fold_dir, 'probe')
    os.makedirs(output_dir, exist_ok=True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows, curves = [], {}
    for layer in LAYERS:
        layer_rows, curve = train_one_layer(
            layer, train, dev, eval_data, eval_split, output_dir, torch.device(device), seed,
            epochs, batch_size, learning_rate,
        )
        rows.extend(layer_rows)
        curves[layer] = curve

    metrics_path = os.path.join(output_dir, 'probe_metrics.csv')
    with open(metrics_path, 'w', newline='', encoding='utf-8') as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, 'probe_training_log.csv'), 'w', newline='', encoding='utf-8') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['layer', 'epoch', 'dev_nll'])
        for layer, curve in curves.items():
            writer.writerows([[layer, epoch, nll] for epoch, nll in curve])
    with open(os.path.join(output_dir, 'probe_config.txt'), 'w', encoding='utf-8') as result_file:
        result_file.write('Probe: one Linear layer + Softmax; frozen DAG-ERC features.\n')
        result_file.write('H0-H4 are the primary comparison and all have the same input dimension.\n')
        result_file.write('Fused is a reference condition with a different input dimension/parameter count.\n')
        result_file.write(f'Epochs: {epochs}; batch size: {batch_size}; learning rate: {learning_rate}; seed: {seed}\n')

    import matplotlib.pyplot as plt
    eval_rows = [row for row in rows if row['split'] == eval_split]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row['layer'] for row in eval_rows], [row['nll'] for row in eval_rows])
    ax.set_ylabel(f'{eval_split.title()} NLL')
    ax.set_title('Linear probe by layer')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'probe_nll_{eval_split}.png'), dpi=160)
    plt.close(fig)
    print(f'Saved probe results to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold_dir', required=True, help='layer_analysis/seedN/foldN')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--eval_split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    train_probes(args.fold_dir, args.output_dir, args.seed, args.epochs,
                 args.batch_size, args.learning_rate, args.device, args.eval_split)


if __name__ == '__main__':
    main()
