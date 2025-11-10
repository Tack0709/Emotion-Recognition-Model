from dataset import MultimodalDAGDataset # 修正版 dataset.py
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import numpy as np

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)

def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    
    # 5クラスのソフトラベルを直接読み込むため、label_vocab は 5クラス に合わせる
    # (注: このファイルは別途作成する必要がある)
    # label_vocab = pickle.load(open('../data/%s/label_vocab_5class.pkl' % (dataset_name), 'rb'))
    
    # 仮のlabel_vocab
    label_vocab = {
        'stoi': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4},
        'itos': {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang', 4: 'oth'}
    }

    return speaker_vocab, label_vocab

def get_multimodal_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab = load_vocab(dataset_name)
    
    print('building datasets..')
    trainset = MultimodalDAGDataset(dataset_name, 'train',  speaker_vocab, args)
    devset = MultimodalDAGDataset(dataset_name, 'dev', speaker_vocab, args)
    
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MultimodalDAGDataset(dataset_name, 'test', speaker_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab