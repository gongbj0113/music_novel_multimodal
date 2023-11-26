from torch.utils.data import Dataset
import pandas as pd
from data.keyword_indexer import KeywordIndexer

TEXT_KEYWORD_DATA_PATH = 'data/csv/text_keyword.csv'

class TextKeywordData(Dataset):
    def __init__(self):
        self.data = pd.read_csv(TEXT_KEYWORD_DATA_PATH)

    def filter(self, keyword_indexer:KeywordIndexer):
        self.data = self.data[self.data['keyword'].isin(keyword_indexer.index_to_keyword)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        keyword = self.data.iloc[idx]['keyword']
        return text, keyword