from dataset import MultimodalDAGDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pickle
import os

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)

def load_vocab(data_dir):
    path = os.path.join(data_dir, 'speaker_vocab.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            speaker_vocab = pickle.load(f)
    else:
        print(f"Warning: {path} not found. Using default speaker vocab.")
        speaker_vocab = {'stoi': {'M': 0, 'F': 1}}
    
    label_vocab = {
        'stoi': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4},
        'itos': {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang', 4: 'oth'}
    }

    return speaker_vocab, label_vocab

# <--- 修正: nma 引数を追加 (デフォルトFalse)
def get_multimodal_loaders(data_dir='output_data', batch_size=32, num_workers=0, args=None, test_session=5, dev_ratio=0.1, nma=False):
    print('building vocab.. ')
    speaker_vocab, label_vocab = load_vocab(data_dir)
    
    print(f'building datasets for Session {test_session} as Test (Dev Ratio: {dev_ratio}, NMA: {nma})...')
    
    # <--- 修正: nma を渡す
    trainset = MultimodalDAGDataset(split='train', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=nma)
    devset = MultimodalDAGDataset(split='dev', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=nma)
    testset = MultimodalDAGDataset(split='test', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir, test_session=test_session, dev_ratio=dev_ratio, nma=nma)
    
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(devset, batch_size=batch_size, sampler=valid_sampler, collate_fn=devset.collate_fn, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab