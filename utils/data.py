import torch
from itertools import cycle
from typing import Iterable, List, Dict, Literal
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from .utils import timer


class LogDataModule:

    def __init__(
        self,
        log_file: str,
        sentence_embedding_model: SentenceTransformer,
        num_fields: int,
        nrows: int = 10**7,
        window_size: int = 20,
        step_size: int = 4,
        batch_size: int = 32,
        train_size: int = 100000,
        random_seed: int = 8,
        **kwargs,
    ):
        self.log_file = log_file
        self.sentence_embedding_model = sentence_embedding_model
        self.num_fields = num_fields
        self.nrows = nrows
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.train_size = train_size
        self.random_seed = random_seed
        self.prepare_data()

    @timer
    def prepare_data(self):
        self.logs, labels = [], []
        with open(self.log_file, 'r') as f:
            for i, row in enumerate(f):
                if i >= self.nrows:
                    break

                fields = row.strip().split(maxsplit=self.num_fields - 1)
                if len(fields) == self.num_fields:
                    labels.append(fields[0] != '-')
                    self.logs.append(' '.join(simple_preprocess(fields[-1], max_len=100)))

        self.embedding = {log: sentence_embedding(log, self.sentence_embedding_model).cpu() for log in set(self.logs)}

        normal, anomaly = [], []
        for i in range(0, len(self.logs) - self.window_size + 1, self.step_size):
            if any(labels[i:i + self.window_size]):
                anomaly.append(i)
            else:
                normal.append(i)

        self.train_x, normal = train_test_split(
            normal,
            train_size=self.train_size,
            random_state=self.random_seed,
        )
        self.eval_x, normal = train_test_split(
            normal,
            train_size=int(self.train_size * 0.1),
            random_state=self.random_seed,
        )
        self.train_y, self.eval_y = [0] * len(self.train_x), [0] * len(self.eval_x)

        val_test_y = [0] * len(normal) + [1] * len(anomaly)
        self.val_x, self.test_x, self.val_y, self.test_y = train_test_split(
            normal + anomaly,
            val_test_y,
            train_size=int(self.train_size * 0.1),
            stratify=val_test_y,
            random_state=self.random_seed,
        )

        self.data_dict = {
            'train': (self.train_x, self.train_y),
            'eval': (self.eval_x, self.eval_y),
            'val': (self.val_x, self.val_y),
            'test': (self.test_x, self.test_y),
        }

    def data_loader(self, mode: Literal['train', 'eval', 'val', 'test']):
        return DataLoader(
            LogSeqDataset(*self.data_dict[mode], self.logs, self.window_size, self.embedding),
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
        )


class LogSeqDataset(Dataset):

    def __init__(
        self,
        indexes: List[int],
        labels: List[int],
        logs: List[str],
        window_size: int,
        embedding: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.indexes = indexes
        self.labels = labels
        self.logs = logs
        self.window_size = window_size
        self.embedding = embedding

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        return torch.stack([self.embedding[log] for log in self.logs[self.indexes[i]:self.indexes[i] + self.window_size]]), self.labels[i]


def sentence_embedding(sentence: str, model: SentenceTransformer):
    with torch.no_grad():
        embedding = model.encode(
            sentence,
            show_progress_bar=False,
            convert_to_tensor=True,
        )

    return embedding


# class CycleZip:

#     def __init__(self, iter1: Iterable, iter2: Iterable):
#         self.iter1 = iter1
#         self.iter2 = iter2

#     def __iter__(self):
#         iter2 = iter(self.iter2)
#         for data1 in self.iter1:
#             try:
#                 data2 = next(iter2)
#             except StopIteration:
#                 iter2 = iter(self.iter2)
#                 data2 = next(iter2)

#             yield data1, data2


class CycleZip:

    def __init__(self, iter1: Iterable, iter2: Iterable):
        self.iter1 = iter1
        self.iter2 = iter2

    def __iter__(self):
        return zip(self.iter1, cycle(self.iter2))


class Chain:

    def __init__(self, *iters: Iterable):
        self.iters = iters

    def __iter__(self):
        for i, iter_ in enumerate(self.iters):
            for data in iter_:
                yield data, i
