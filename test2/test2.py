import sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))
import torch
import torch.nn as nn
import torch.optim as optim
from tempfile import mkdtemp
import importlib
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, OutputFile, OutputDirectory, InputFile, InputDirectory


@dsl.module(version='0.0.3')
def test2(
        trained_model_dir: InputDirectory(type='AnyDirectory') = None,
        test_data_dir: InputDirectory(type='AnyDirectory') = None
):
    # /mnt/batch/tasks/shared/LS_root/jobs/fundamental/azureml/72e078e0-0677-4e13-9ab0-b50c75b6faba/mounts/workspaceblobstore/azureml/f9272384-1f23-49b1-8218-2cce8297cdc3/Training_Data_Output
    print('trained_model_dir:', trained_model_dir)
    # /mnt/batch/tasks/shared/LS_root/jobs/fundamental/azureml/72e078e0-0677-4e13-9ab0-b50c75b6faba/mounts/workspaceblobstore/azureml/f9272384-1f23-49b1-8218-2cce8297cdc3/Validation_Data_Output
    print('test_data_dir:', test_data_dir)
    print('trained_model_dir目录下的文件包括:', os.listdir(trained_model_dir))
    # Datasets
    test_dataset = YelpLoader(path_to_csv=os.path.join(test_data_dir, 'test_data.csv'))

    # DataLoaders
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Length of Test Dataset is :{0}".format(len(test_dataloader)))

    # Model
    path = os.path.join(trained_model_dir, 'BestModel')
    best_HAN = torch.load(f=path)

    # Run Test Dataset after Training
    print("Predicting on Test Set")
    true_test_labels, pred_test_labels = model_test(best_HAN, test_dataloader)
    print(classification_report(y_true=true_test_labels, y_pred=pred_test_labels))


class HanModel(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 32,
                 bidirectional: bool = True,
                 layers: int = 2,
                 padding_idx: int = 0,
                 class_size: int = 5,
                 randomize_init_hidden: bool = True
                 ):

        super(HanModel, self).__init__()
        # Model Properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.layers = layers
        self.directions = 2 if self.bidirectional else 1
        self.class_size = class_size
        self.randomize_init_hidden = randomize_init_hidden

        # Useful Consts
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        # Model Layers
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

        self.elmo_embed = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)

        self.word_gru = nn.GRU(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               num_layers=self.layers,
                               batch_first=False
                               )
        self.word_linear = nn.Linear(in_features=self.directions * self.hidden_dim,
                                     out_features=self.directions * self.hidden_dim,
                                     bias=True)
        self.word_context = nn.Linear(in_features=self.directions * self.hidden_dim,
                                      out_features=1,
                                      bias=False)

        self.sent_gru = nn.GRU(input_size=self.directions * self.hidden_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               num_layers=self.layers,
                               batch_first=False
                               )
        self.sent_linear = nn.Linear(in_features=self.directions * self.hidden_dim,
                                     out_features=self.directions * self.hidden_dim,
                                     bias=True)
        self.sent_context = nn.Linear(in_features=self.directions * self.hidden_dim,
                                      out_features=1,
                                      bias=False)

        self.fc1 = nn.Linear(self.hidden_dim * self.directions, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.class_size)

    def init_hidden(self, batch_size: int = 1):
        if self.randomize_init_hidden:
            init_hidden = torch.randn(self.layers * self.directions, batch_size, self.hidden_dim)
        else:
            init_hidden = torch.zeros(self.layers * self.directions, batch_size, self.hidden_dim)
        return init_hidden

    def forward(self, abstracts_input_packet):
        abstracts_output_packet = []
        for curr_abstract_sentences, curr_abstract_lens, curr_abstract_unsorted_ix in abstracts_input_packet:
            abstracts_output_packet.append(
                self._forward_single_abstract(curr_abstract_sentences, curr_abstract_lens,
                                              curr_abstract_unsorted_ix))
        return abstracts_output_packet

    def _forward_single_abstract(self, sentences, sentences_lens, sentence_unsorted_ix):
        # Extract Batch Size
        sentences_batch_size = len(sentences)

        # Embed the Words with Elmo
        character_ids = batch_to_ids(sentences)
        sentences_embedded = self.elmo_embed(character_ids)['elmo_representations'][0]

        # Reshape the input ( for packing )
        sentences_embedded = torch.transpose(sentences_embedded, 0, 1)
        # Pack the Words
        sentences_packed = nn.utils.rnn.pack_padded_sequence(
            input=sentences_embedded,
            lengths=sentences_lens.cpu().numpy(),
            batch_first=False,
        )
        # Run Words through Word_GRU
        init_batch_hidden = self.init_hidden(sentences_batch_size)
        word_gru_op_packed, _ = self.word_gru(sentences_packed, init_batch_hidden)

        # Unpack your packed_output
        word_gru_output, input_sizes = nn.utils.rnn.pad_packed_sequence(word_gru_op_packed, batch_first=False)
        word_gru_output = word_gru_output.view(sentences_lens.max(), sentences_batch_size, self.directions,
                                               self.hidden_dim)

        # Reshape the Output
        word_gru_output = torch.transpose(word_gru_output, 0, 1)

        # Unsort the Sentences to Original Order
        word_gru_output = word_gru_output[sentence_unsorted_ix]
        word_gru_output_lens = sentences_lens[sentence_unsorted_ix]

        # Joining the Hidden Layers in the Output
        word_gru_output = [torch.reshape(word_gru_output[sent_ix, :word_gru_output_len, :, :],
                                         (1, word_gru_output_len, self.directions * self.hidden_dim)).squeeze() for
                           sent_ix, word_gru_output_len in
                           zip(range(word_gru_output.shape[0]), word_gru_output_lens)]

        # Word Attn
        word_attns = [self.softmax(self.word_context(self.tanh(self.word_linear(sentence_hiddens)))) for
                      sentence_hiddens in word_gru_output]

        # Hack to take care of Single Word Sentences
        word_attns = [
            sentence_word_attn.unsqueeze(dim=1) if len(sentence_word_attn.shape) == 1 else sentence_word_attn
            for sentence_word_attn in word_attns]

        # Sentence Repr
        sentence_repr = torch.stack(
            [sum(sentence_word_attn * sentence_hiddens) for sentence_word_attn, sentence_hiddens in
             zip(word_attns, word_gru_output)])

        # Prep for Sent Gru ( Add Batch Dimension at Dim = 1)
        sentence_repr.unsqueeze_(1)

        init_batch_hidden = self.init_hidden()
        sent_gru_output, _ = self.sent_gru(sentence_repr, init_batch_hidden)

        # Unsqueeze the Sent Gru Output
        sent_gru_output.squeeze_(1)

        sent_attns = self.softmax(self.sent_context(self.tanh(self.sent_linear(sent_gru_output))))

        # Document Repr
        doc_repr = sum(sent_attns * sent_gru_output)

        # Run through Final Fully Connected Layer
        output_unnormalized = self.fc1(doc_repr)
        output_unnormalized = self.fc2(output_unnormalized)

        return output_unnormalized, sent_attns, word_attns


class YelpLoader(Dataset):
    def __init__(self, path_to_csv):
        self.data_tuples = []
        df = pd.read_csv(filepath_or_buffer=path_to_csv, sep='\t', engine='python')
        df['text'] = df['text'].apply(str.lower)
        df['stars'] = df['stars'].apply(lambda x: x - 1)
        self.data_tuples = [(r['text'], r['stars']) for _, r in df.iterrows()]
        self.ctgry = sorted(df['stars'].unique())

    def __getitem__(self, index):
        return self.data_tuples[index]

    def __len__(self):
        return len(self.data_tuples)


def prep_each_document(document: str):
    unsorted_sent_words = [[token.text for token in sent] for sent in nlp(document).sents]
    unsorted_sent_lens = torch.tensor([len(sent_words) for sent_words in unsorted_sent_words])

    _, sorted_sent_ixs = unsorted_sent_lens.sort(dim=0, descending=True)
    _, unsorted_sent_ixs = sorted_sent_ixs.sort(dim=0, descending=False)

    sorted_sent_words = []
    unsorted_sent_packet = torch.zeros(len(unsorted_sent_lens), max(unsorted_sent_lens), dtype=torch.long)
    for sorted_sent_ix in sorted_sent_ixs:
        sorted_sent_words.append(unsorted_sent_words[sorted_sent_ix])

    sorted_sent_lens = torch.tensor([len(sent_words) for sent_words in sorted_sent_words])

    return sorted_sent_words, sorted_sent_lens, unsorted_sent_ixs


def model_train(model, model_criterion, model_optimizer, dataloader):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        model_optimizer.zero_grad()
        batch_texts, batch_labels = batch
        batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
        batch_output_packets = model(batch_input_packets)
        batch_output_unnormalized = torch.stack(
            [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
        loss = model_criterion(input=batch_output_unnormalized,
                               target=batch_labels)
        loss.backward()
        model_optimizer.step()
        epoch_loss += loss.item() / len(batch_input_packets)
    return model, model_criterion, model_optimizer, epoch_loss


def model_evaluate(model, model_criterion, dataloader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_texts, batch_labels = batch
            batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
            batch_output_packets = model(batch_input_packets)
            batch_output_unnormalized = torch.stack(
                [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
            loss = model_criterion(input=batch_output_unnormalized,
                                   target=batch_labels)
            epoch_loss += loss.item() / len(batch_input_packets)
    return epoch_loss

def model_test(model, dataloader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    epoch_loss = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch_texts, batch_labels = batch
            batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
            batch_output_packets = model(batch_input_packets)
            batch_output_unnormalized = torch.stack(
                [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
            batch_scores_normalized = softmax(batch_output_unnormalized)

            true_labels.append(batch_labels)
            pred_labels.append(torch.argmax(batch_scores_normalized, dim=1))
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    return true_labels, pred_labels


if __name__ == "__main__":
    ModuleExecutor(test2).execute(sys.argv)
