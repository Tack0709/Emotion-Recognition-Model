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
    datapath = os.path.join('../data', dataset_name)
    speaker_vocab = pickle.load(open(os.path.join(datapath, 'speaker_vocab.pkl'), 'rb'))
    
    # 5クラス（neu, hap, sad, ang, oth） の語彙を定義
    label_vocab = {
        'stoi': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4},
        'itos': {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang', 4: 'oth'}
    }
    # (注: 'xxx' は dataset.py で除外されるため、ここには不要)

    return speaker_vocab, label_vocab

# collate_fnがNoneを返す可能性に対処するラッパー
def collate_fn_wrapper(batch):
    # dataset.py の collate_fn を呼び出す
    # (data[0] は MultimodalDAGDataset のインスタンス)
    processed_batch = batch[0][0].collate_fn(batch) 
    
    # collate_fn が空のダイアログなどでNoneを返した場合のフィルタリング
    if processed_batch[0] is None:
        return None
    return processed_batch

def filter_none_collate(batch):
    """
    DataLoaderに渡すためのcollate関数。
    dataset.pyのcollate_fnを呼び出し、Noneが返された場合はフィルタリングする。
    """
    # バッチ内のデータを使ってcollate_fnを呼び出す (datasetインスタンスが必要)
    if not batch:
        return None
    
    # バッチの最初の要素から dataset インスタンスを取得
    # (注: DataLoaderは __getitem__ の戻り値のリストを batch として渡す)
    # (注2: この実装は DataLoader が dataset インスタンスにアクセスできない前提)
    # (より良い実装は、 DataLoader 生成時に dataset.collate_fn を直接渡すこと)
    
    # この設計では、dataloader.py が dataset.py の実装を知らない前提で
    # run.py で collate_fn を渡すのが最善。
    
    # --> run.py で collate_fn を渡すように変更します (このファイルは変更不要)
    pass


def get_multimodal_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab = load_vocab(dataset_name)
    
    print('building datasets..')
    # 修正: MultimodalDAGDataset を使用
    trainset = MultimodalDAGDataset(dataset_name, 'train',  speaker_vocab, args)
    devset = MultimodalDAGDataset(dataset_name, 'dev', speaker_vocab, args)
    
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    # collate_fn は dataset インスタンスのメソッドを指定する
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn, #
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn, #
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MultimodalDAGDataset(dataset_name, 'test', speaker_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn, #
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab