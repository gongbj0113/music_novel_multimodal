from typing import Optional
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

import sys
if '../' not in sys.path:
    sys.path.append('../')

from data.text_keyword_data import TextKeywordData
from data.keyword_indexer import KeywordIndexer


class BERTDataset(Dataset):
    def __init__(self, text_keyword_dataset, keyword_indexer:KeywordIndexer, bert_tokenizer, max_len, pad, pair):
        """
        A PyTorch dataset for BERT input data.

        Args:
            text_keyword_dataset (list): List of tuples containing text and keyword pairs.
            keyword_indexer (KeywordIndexer): KeywordIndexer object for converting keywords to indices.
            bert_tokenizer: BERT tokenizer for tokenizing the text.
            max_len (int): Maximum sequence length.
            pad (bool): Whether to pad the sequences.
            pair (bool): Whether the input is a pair of sentences.
        """
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[0]]) for i in text_keyword_dataset]
        self.labels = [np.int32(keyword_indexer.transform_text_to_index(i[1].strip())) for i in text_keyword_dataset]

    def __getitem__(self, i):
        """
        Get the i-th item from the dataset.

        Args:
            i (int): Index of the item.

        Returns:
            tuple: Tuple containing the tokenized sentence, attention mask, segment IDs, and label.
        """
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None, params=None):
        """
        BERT classifier model.

        Args:
            bert: Pretrained BERT model.
            hidden_size (int): Size of the hidden layer.
            num_classes (int): Number of output classes.
            dr_rate: Dropout rate.
            params: Additional parameters.
        """
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        """
        Generate attention mask.

        Args:
            token_ids: Tokenized input.
            valid_length: Length of the valid input.

        Returns:
            torch.Tensor: Attention mask.
        """
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        """
        Forward pass of the model.

        Args:
            token_ids: Tokenized input.
            valid_length: Length of the valid input.
            segment_ids: Segment IDs.

        Returns:
            torch.Tensor: Output logits.
        """
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

class TextKeywordClassifier:
    def __init__(self, keyword_indexer:KeywordIndexer):
        """
        Text keyword classifier.

        Args:
            keyword_indexer (KeywordIndexer): KeywordIndexer object for converting keywords to indices.
        """
        self.keyword_indexer = keyword_indexer

        self.device = torch.device("cuda:0")
        self.model, self.vocab = get_pytorch_kobert_model()
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.model = BERTClassifier(self.model, num_classes=keyword_indexer.get_num_keywords(), dr_rate=0.5).to(self.device)
        self.loaded = False

        self.max_len = 300
        self.batch_size = 32

    def load(self):
        """
        Load the trained model.
        """
        self.model.load_state_dict(torch.load('model/save/text_keyword_classifier.pt', map_location=self.device))
        self.model.eval()

    def train(self, text_keyword_data:TextKeywordData):
        """
        Train the text keyword classifier.

        Args:
            text_keyword_data (TextKeywordData): TextKeywordData object containing the training data.
        """
        train, test = train_test_split(text_keyword_data, test_size=0.2, shuffle=True, random_state=0)
        
        warmup_ratio = 0.1
        num_epochs = 100
        max_grad_norm = 1
        log_interval = 200
        learning_rate = 5e-5

        train_dataset = BERTDataset(train, self.keyword_indexer, self.tok, self.max_len, True, False)
        test_dataset = BERTDataset(test, self.keyword_indexer, self.tok, self.max_len, True, False)

        # num_workers : how many subprocesses to use for data loading
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=5)

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)  # optimizer
        # try different optimizer
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

        loss_fn = nn.CrossEntropyLoss()  # loss function

        t_total = len(train_loader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


        def calc_accuracy(X, Y):
            max_vals, max_indices = torch.max(X, 1)
            train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
            return train_acc
        
        train_acc_history = []
        test_acc_history = []
        
        for e in range(num_epochs):
            train_acc = 0.0
            test_acc = 0.0

            # Train
            self.model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_loader)):
                optimizer.zero_grad()

                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length = valid_length
                # print(label)
                label = label.long().to(self.device)

                out = self.model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                train_acc += calc_accuracy(out, label)
                if batch_id % log_interval == 0:
                    print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                            train_acc / (batch_id + 1)))
            train_acc_history.append(train_acc / (batch_id + 1))
            print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

            # Evaluation
            self.model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length = valid_length
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)
            test_acc_history.append(test_acc / (batch_id + 1))
            print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

        self.loaded = True
        torch.save(self.model.state_dict(), 'model/save/text_keyword_classifier.pt')

        import matplotlib.pyplot as plt
        # Plot and save the graph
        plt.plot(range(1, num_epochs + 1), train_acc_history, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), test_acc_history, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_graph.png')
        plt.show()

        return test

    def predict(self, text) -> Optional[str]:
        """
        Predict the keyword for the given text.

        Args:
            text (str): Input text.

        Returns:
            Optional[str]: Predicted keyword or None if there are some problems.
        """
        # If not loaded, load the model
        if not self.loaded:
            self.load()
        
        # Predict
        data = [text, self.keyword_indexer.transform_index_to_text(0)]
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, self.keyword_indexer, self.tok, self.max_len, True, False)
        test_loader = DataLoader(another_test, batch_size=self.batch_size, num_workers=0)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length= valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)

            test_eval=[]
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                return self.keyword_indexer.transform_index_to_text(np.argmax(logits))
        
        return None