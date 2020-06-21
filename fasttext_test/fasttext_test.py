import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, InputDirectory, OutputDirectory

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)


@dsl.module(
    name="FastText Test",
    version='0.0.7',
    description='Test the trained FastText model'
)
def fasttext_test(
        trained_model_dir: InputDirectory(type='AnyDirectory') = None,
        test_data_dir: InputDirectory(type='AnyDirectory') = None,
        char2index_dir: InputDirectory(type='AnyDirectory') = None
):
    print('============================================')
    print('test_data_dir:', test_data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len_ = 38
    path = os.path.join(test_data_dir, 'test.txt')
    test_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)

    test_iter = DataIter(test_samples)

    path = os.path.join(trained_model_dir, 'BestModel')
    model = torch.load(f=path)

    test(model, test_iter, device)
    print('\n============================================')


def get_vocab(char2index_dir):
    c2i = json.load(open(char2index_dir, 'r', encoding='utf-8'))
    return c2i


def test(model, test_iter=None, device=None):
    acc_, loss_ = eval(model, test_iter, device)
    str_ = f"test acc:{acc_:.4f}"
    sys.stdout.write(str_)
    sys.stdout.flush()


def eval(model, data_iter, device):
    model.eval()
    with torch.no_grad():
        acc_list = []
        loss_list = []

        for x, y in data_iter:
            dev_x_ = torch.LongTensor(x).to(device)
            dev_y_ = torch.LongTensor(y).to(device)
            outputs = model(dev_x_)
            p_ = torch.max(outputs.data, 1)[1].cpu().numpy()
            acc_ = metrics.accuracy_score(y, p_)
            loss_ = torch.nn.CrossEntropyLoss()(outputs, dev_y_)
            acc_list.append(acc_)
            loss_list.append(loss_.cpu().data.numpy())
        return np.mean(acc_list), np.mean(loss_list)


class FastText(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_class,
                 embed_dim=128,
                 ):
        super(FastText, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.fc = nn.Linear(in_features=embed_dim,
                            out_features=n_class)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = F.max_pool2d(x, (x.shape[-2], 1))  # [batch, 1, embed_dim]
        x = x.squeeze()  # [batch, embed_dim]
        x = self.fc(x)  # [batch, n_class]
        x = torch.sigmoid(x)  # [batch, n_class]
        return x


def shuffle_samples(samples):
    samples = np.array(samples)
    shffle_index = np.arange(len(samples))
    np.random.shuffle(shffle_index)
    samples = samples[shffle_index]
    return samples


def load_dataset(file_path='', max_len=38, char2index_dir=''):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [item.strip() for item in texts if len(item) > 0]
    c2i = get_vocab(char2index_dir)
    pad_id = c2i.get('[PAD]', 0)
    samples = []
    for line in tqdm(texts, desc="character to index"):
        line_s = line.split('\t')
        if len(line_s) < 2:
            continue
        context, label = line_s[0], line_s[1]
        line_data = ([c2i.get(c, 1) for c in context]) + [pad_id] * (max_len - len(context))
        line_data = line_data[:max_len]
        samples.append((line_data, int(label)))
    samples = np.array(samples)
    return samples


class DataIter(object):
    def __init__(self, samples, batch_size=32, shuffle=True):
        if shuffle:
            samples = shuffle_samples(samples)
        self.samples = samples
        self.batch_size = batch_size
        self.n_batches = len(samples) // self.batch_size
        self.residue = (len(samples) % self.n_batches != 0)
        self.index = 0

    def split_samples(self, sub_samples):
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        return np.array(b_x), np.array(b_y)

    def __next__(self):
        if (self.index == self.n_batches) and (self.residue is True):
            sub_samples = self.samples[self.index * self.batch_size: len(self.samples)]
            self.index += 1
            return self.split_samples(sub_samples)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            sub_samples = self.samples[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self.split_samples(sub_samples)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


if __name__ == '__main__':
    ModuleExecutor(fasttext_test).execute(sys.argv)
