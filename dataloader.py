from dataset import MultimodalDAGDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pickle
import os

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)

def load_vocab(data_dir, args=None):
    path = os.path.join(data_dir, 'speaker_vocab.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            speaker_vocab = pickle.load(f)
    else:
        print(f"Warning: {path} not found. Using default speaker vocab.")
        speaker_vocab = {'stoi': {'M': 0, 'F': 1}}
    
    if args and getattr(args, 'raw_emotions', False):
        raw_vocab_path = os.path.join(data_dir, 'emotion_vocab_raw.pkl')
        if not os.path.exists(raw_vocab_path):
            raise FileNotFoundError(
                f'{raw_vocab_path} not found. Re-run create_label.py to generate raw-tag labels.'
            )
        with open(raw_vocab_path, 'rb') as f:
            label_vocab = pickle.load(f)
    else:
        label_vocab = {
            'stoi': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4},
            'itos': {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang', 4: 'oth'},
            'unknown_id': 5,
        }

    return speaker_vocab, label_vocab

def get_multimodal_loaders(data_dir='output_data', batch_size=32, num_workers=0, args=None, test_session=5, dev_ratio=0.1, nma=False, train_nma=None, test_nma=None, preserve_order=False, splits=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab = load_vocab(data_dir, args)
    
    # 引数の整理: 個別の指定がない場合は nma (全体設定) に従う
    if train_nma is None: train_nma = nma
    if test_nma is None: test_nma = nma

    print(f'building datasets for Session {test_session} as Test (Dev Ratio: {dev_ratio}).')
    print(f'  [Config] Train NMA: {train_nma} | Test NMA: {test_nma}')
    
    # split='train', 'dev' には train_nma を適用
    trainset = MultimodalDAGDataset(split='train', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=train_nma)
    devset = MultimodalDAGDataset(split='dev', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=train_nma)
    
    # 探索分析ではTestデータをロードしないため、必要なsplitだけ生成できる。
    # 既存の学習コードとの互換性のため、未指定時は従来どおり全splitを作る。
    requested_splits = set(splits or ('train', 'dev', 'test'))
    if not requested_splits.issubset({'train', 'dev', 'test'}):
        raise ValueError(f'Unsupported data splits: {sorted(requested_splits)}')
    testset = None
    if 'test' in requested_splits:
        # split='test' には test_nma を適用
        testset = MultimodalDAGDataset(split='test', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=test_nma)
    
    train_sampler = None if preserve_order else get_train_valid_sampler(trainset)
    valid_sampler = None if preserve_order else get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(devset, batch_size=batch_size, sampler=valid_sampler, collate_fn=devset.collate_fn, num_workers=num_workers)
    test_loader = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab
