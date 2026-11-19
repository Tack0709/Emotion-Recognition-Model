"""Extract fixed DAG-ERC intermediate features for one saved checkpoint."""
import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from dataloader import get_multimodal_loaders
from model import DAGERC_multimodal


def verify_reconstructed_split(train_loader, dev_loader, seed, test_session, dev_ratio):
    train_dataset = train_loader.dataset
    dev_dataset = dev_loader.dataset
    if train_dataset.order.keys() != dev_dataset.order.keys():
        raise ValueError('Train and Dev datasets loaded different dialogue metadata')

    test_prefix = f'Ses0{test_session}'
    candidate_sessions = sorted(
        session_id for session_id in train_dataset.order
        if not session_id.startswith(test_prefix)
    )
    random.Random(seed).shuffle(candidate_sessions)
    dev_size = int(len(candidate_sessions) * dev_ratio)
    expected_dev = candidate_sessions[:dev_size]
    expected_train = candidate_sessions[dev_size:]
    actual_train = list(train_dataset.session_list)
    actual_dev = list(dev_dataset.session_list)
    if actual_train != expected_train or actual_dev != expected_dev:
        raise ValueError(
            'Reconstructed Train/Dev dialogue split does not match the checkpoint seed, '
            'test session, and dev_ratio')
    if set(actual_train) & set(actual_dev):
        raise ValueError('Train and Dev dialogue sessions overlap')
    if set(actual_train) | set(actual_dev) != set(candidate_sessions):
        raise ValueError('Train and Dev dialogue sessions do not cover the non-Test sessions')
    return actual_train, actual_dev


def entropy_from_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = probabilities.sum()
    if not np.isfinite(probabilities).all() or total <= 0:
        return np.nan
    probabilities = probabilities / total
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        return torch.load(path, map_location='cpu')


def extract_split(model, loader, device, output_path, metadata, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f'{output_path} already exists; pass --overwrite to replace it')

    layer_names = [f'H{i}' for i in range(model.args.gnn_layers + 1)]
    chunks = {name: [] for name in layer_names + ['Fused', 'pre_mlp', 'true_soft', 'pred_prob']}
    rows = {name: [] for name in (
        'dialog_id', 'utterance_id', 'turn_index', 'speaker', 'text', 'true_entropy'
    )}

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if batch[0] is None:
                continue
            features_text, features_audio = batch[0].to(device), batch[1].to(device)
            labels_soft = batch[2]
            adj, s_mask, s_mask_onehot = batch[3].to(device), batch[4].to(device), batch[5].to(device)
            lengths = batch[6]

            predictions, intermediate = model(
                features_text, features_audio, adj, s_mask, s_mask_onehot,
                lengths, return_intermediates=True,
            )
            valid = (labels_soft >= 0).all(dim=2) & (labels_soft.sum(dim=2) > 0)
            valid_indices = torch.nonzero(valid, as_tuple=False).tolist()
            prediction_cpu = predictions.detach().cpu().numpy()
            labels_cpu = labels_soft.numpy()
            intermediate_cpu = {
                name: values.detach().cpu().numpy() for name, values in intermediate.items()
            }

            for batch_index, turn_index in valid_indices:
                for name in layer_names + ['Fused', 'pre_mlp']:
                    chunks[name].append(intermediate_cpu[name][batch_index, turn_index])
                true_distribution = labels_cpu[batch_index, turn_index].astype(np.float32)
                predicted_distribution = prediction_cpu[batch_index, turn_index].astype(np.float32)
                chunks['true_soft'].append(true_distribution)
                chunks['pred_prob'].append(predicted_distribution)
                rows['dialog_id'].append(str(batch[10][batch_index]))
                rows['utterance_id'].append(str(batch[11][batch_index][turn_index]))
                rows['turn_index'].append(int(batch[13][batch_index][turn_index]))
                rows['speaker'].append(str(batch[12][batch_index][turn_index]))
                rows['text'].append(str(batch[8][batch_index][turn_index]))
                rows['true_entropy'].append(entropy_from_probabilities(true_distribution))

    if not rows['utterance_id']:
        raise ValueError(f'No valid soft-label utterances found in {metadata["split"]} split')
    row_count = len(rows['utterance_id'])
    if any(len(values) != row_count for values in chunks.values()):
        raise RuntimeError('Layer features, labels, predictions, and metadata lost row alignment')

    arrays = {key: np.asarray(values) for key, values in chunks.items()}
    arrays.update({key: np.asarray(values) for key, values in rows.items()})
    arrays['fold'] = np.asarray(metadata['fold'])
    arrays['seed'] = np.asarray(metadata['seed'])
    arrays['split'] = np.asarray(metadata['split'])
    arrays['class_names'] = np.asarray(metadata['class_names'], dtype=object)
    arrays['checkpoint'] = np.asarray(metadata['checkpoint'])
    np.savez_compressed(output_path, **arrays)
    print(f"Saved {len(rows['utterance_id'])} utterances to {output_path}")


def extract_checkpoint(checkpoint_path, data_dir=None, output_dir='layer_analysis', overwrite=False,
                       expected_seed=None, expected_fold=None, splits=('train', 'dev')):
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = checkpoint.get('args')
    if not isinstance(saved_args, dict):
        raise ValueError('Checkpoint does not contain saved experiment arguments')
    args = SimpleNamespace(**saved_args)
    if expected_seed is not None and int(args.seed) != expected_seed:
        raise ValueError(f'Checkpoint seed {args.seed} does not match requested seed {expected_seed}')
    checkpoint_fold = int(checkpoint.get('fold', getattr(args, 'test_session', 0)))
    if expected_fold is not None and checkpoint_fold != expected_fold:
        raise ValueError(f'Checkpoint fold {checkpoint_fold} does not match requested fold {expected_fold}')
    if getattr(args, 'simple_nn', False) or getattr(args, 'standard_gnn', False) or getattr(args, 'edl_r2', False):
        raise ValueError('Layer extraction currently supports standard DAG-ERC checkpoints only')
    if getattr(args, 'raw_emotions', False):
        raise ValueError('Layer analysis expects the aggregated five-class labels')
    if not getattr(args, 'full', False):
        raise ValueError('Layer analysis requires a checkpoint trained with --full')
    if getattr(args, 'nodal_att_type', None) is not None:
        raise ValueError('Layer analysis requires a Nodal Attention=None checkpoint')
    if args.modality != 'multimodal':
        raise ValueError('Layer analysis requires a multimodal checkpoint')
    if args.gnn_layers != 4 or args.hidden_dim != 300:
        raise ValueError('Expected the specified four-layer, hidden-dimension-300 DAG-ERC checkpoint')

    args.data_dir = data_dir or getattr(args, 'data_dir', None) or 'output_data'
    args.test_session = checkpoint_fold
    args.cuda = torch.cuda.is_available() and not getattr(args, 'no_cuda', False)
    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    device = torch.device('cuda' if args.cuda else 'cpu')

    train_loader, dev_loader, test_loader, _, label_vocab = get_multimodal_loaders(
        data_dir=args.data_dir,
        batch_size=getattr(args, 'batch_size', 16),
        num_workers=0,
        args=args,
        test_session=args.test_session,
        dev_ratio=getattr(args, 'dev_ratio', 0.2),
        train_nma=True,
        test_nma=True,
        preserve_order=True,
        splits=splits,
    )
    train_sessions, dev_sessions = verify_reconstructed_split(
        train_loader, dev_loader, int(args.seed), checkpoint_fold,
        float(getattr(args, 'dev_ratio', 0.2)),
    )
    model = DAGERC_multimodal(args, len(label_vocab['itos']))
    if len(label_vocab['itos']) != 5:
        raise ValueError(f'Expected 5 aggregated classes, found {len(label_vocab["itos"])}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    fold_dir = os.path.join(output_dir, f"seed{args.seed}", f"fold{args.test_session}", 'features')
    os.makedirs(fold_dir, exist_ok=True)
    existing_outputs = [os.path.join(fold_dir, f'{split}_features.npz') for split in splits
                        if os.path.exists(os.path.join(fold_dir, f'{split}_features.npz'))]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            'Layer feature files already exist; refusing to overwrite: ' + ', '.join(existing_outputs)
        )
    manifest_path = os.path.join(os.path.dirname(fold_dir), 'split_manifest.json')
    manifest = {
        'seed': int(args.seed), 'fold': int(args.test_session),
        'test_session': int(args.test_session), 'dev_ratio': float(getattr(args, 'dev_ratio', 0.2)),
        'split_source': 'reconstructed from checkpoint args and current deterministic dataset split',
        'train_sessions': train_sessions, 'dev_sessions': dev_sessions,
        'test_split_loaded': 'test' in splits,
    }
    if os.path.isfile(manifest_path) and not overwrite:
        with open(manifest_path, encoding='utf-8') as source:
            previous_manifest = json.load(source)
        manifest_keys = ('seed', 'fold', 'test_session', 'dev_ratio', 'train_sessions',
                         'dev_sessions', 'split_source')
        if any(previous_manifest.get(key) != manifest[key] for key in manifest_keys):
            raise ValueError(f'Existing split manifest does not match reconstructed split: {manifest_path}')
    else:
        with open(manifest_path, 'w', encoding='utf-8') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
        print(f'Saved reconstructed Train/Dev split manifest to {manifest_path}')
    class_names = [label_vocab['itos'][i] for i in range(len(label_vocab['itos']))]
    loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}
    for split_name in splits:
        loader = loaders[split_name]
        output_path = os.path.join(fold_dir, f'{split_name}_features.npz')
        extract_split(
            model, loader, device, output_path,
            {
                'fold': args.test_session,
                'seed': args.seed,
                'split': split_name,
                'class_names': class_names,
                'checkpoint': os.path.abspath(checkpoint_path),
            },
            overwrite=overwrite,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default='layer_analysis')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--expected_seed', type=int, default=None)
    parser.add_argument('--expected_fold', type=int, default=None)
    parser.add_argument('--splits', nargs='+', choices=['train', 'dev', 'test'],
                        default=['train', 'dev'])
    args = parser.parse_args()
    extract_checkpoint(args.checkpoint, args.data_dir, args.output_dir, args.overwrite,
                       args.expected_seed, args.expected_fold, args.splits)


if __name__ == '__main__':
    main()
