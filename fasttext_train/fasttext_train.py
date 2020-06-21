import os
import sys
import json
import time
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
    name="FastText Train",
    version='0.0.8',
    description='Train the FastText model. You could adjust the hyperparameters conveniently'
)
def fasttext_train(
        trained_model_dir: OutputDirectory(type='AnyDirectory'),
        training_data_dir: InputDirectory(type='AnyDirectory') = None,
        validation_data_dir: InputDirectory(type='AnyDirectory') = None,
        char2index_dir: InputDirectory(type='AnyDirectory') = None,
        epochs=2,
        batch_size=32,
        learning_rate=0.0005,
        embedding_dim=128

):
    print('============================================')
    print('training_data_dir:', training_data_dir)
    print('validation_data_dir:', validation_data_dir)
    c2i = get_vocab(char2index_dir)
    class_ = get_classs()
    max_len_ = 38
    n_class_ = len(class_)
    vocab_size_ = len(c2i)
    stop_patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join(training_data_dir, 'train.txt')
    train_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)
    path = os.path.join(validation_data_dir, 'dev.txt')
    dev_samples = load_dataset(file_path=path, max_len=max_len_, char2index_dir=char2index_dir)

    train_iter = DataIter(train_samples, batch_size)
    dev_iter = DataIter(dev_samples, batch_size)

    model = FastText(vocab_size=vocab_size_, n_class=n_class_, embed_dim=embedding_dim)
    start = time.time()
    train(model,
          trained_model_dir,
          train_iter,
          dev_iter=dev_iter,
          epochs=epochs,
          learning_rate=learning_rate,
          stop_patience=stop_patience,
          device=device)
    end = time.time()
    print('\nspent time: %.2f sec' % (end - start))
    print('============================================')


class FastText(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_class,
                 embed_dim,
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


def train(model,
          trained_model_dir: OutputDirectory(type='AnyDirectory'),
          train_iter,
          dev_iter=None,
          epochs=20,
          learning_rate=0.0001,
          stop_patience=3,
          device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()

    min_loss_epoch = (None, None)
    stop_flag = False

    model_name = model._get_name()
    tip_str = f"\n{model_name} start training....."
    print(tip_str)

    for epoch in range(epochs):
        loss_value_list = []
        total_iter = len(train_iter)
        for i, (x_batch, y_batch) in enumerate(train_iter):
            x_batch = torch.LongTensor(x_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            outputs = model(x_batch)
            optimizer.zero_grad()
            loss_value = loss(outputs, y_batch)
            loss_value.backward()
            optimizer.step()

            loss_value_list.append(loss_value.cpu().data.numpy())

            str_ = f"{model_name} epoch:{epoch + 1}/{epochs} step:{i + 1}/{total_iter} mean_loss:{np.mean(loss_value_list): .4f}"
            sys.stdout.write('\r' + str_)
            sys.stdout.flush()

            if (i + 1) == total_iter and dev_iter is not None:
                acc_, loss_ = eval(model, dev_iter, device)
                str_ = f" validation loss:{loss_:.4f}  acc:{acc_:.4f}"
                sys.stdout.write(str_)
                sys.stdout.flush()
                print()

                model.train()

                if (min_loss_epoch[0] is None) or (min_loss_epoch[0] > loss_):
                    min_loss_epoch = (loss_, epoch)
                    os.makedirs(trained_model_dir, exist_ok=True)
                    path = os.path.join(trained_model_dir, "BestModel")
                    torch.save(obj=model, f=path)
                else:
                    if (epoch - min_loss_epoch[1]) >= stop_patience:
                        stop_flag = True
                        break

        if stop_flag is True:
            break


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


class DataIter(object):
    def __init__(self, samples, batch_size, shuffle=True):
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


def shuffle_samples(samples):
    samples = np.array(samples)
    shffle_index = np.arange(len(samples))
    np.random.shuffle(shffle_index)
    samples = samples[shffle_index]
    return samples


def get_vocab(char2index_dir):
    c2i = json.load(open(char2index_dir, 'r', encoding='utf-8'))
    return c2i


def get_classs():
    res = {'finance': 0,
           'realty': 1,
           'stocks': 2,
           'education': 3,
           'science': 4,
           'society': 5,
           'politics': 6,
           'sports': 7,
           'game': 8,
           'entertainment': 9}
    return res


if __name__ == '__main__':
    ModuleExecutor(fasttext_train).execute(sys.argv)
