from torch.utils.data import Dataset
import pandas as pd

TEXT_KEYWORD_DATA_PATH = 'data/csv/text_keyword.csv'

class TextKeywordData(Dataset):
    def __init__(self):
        self.data = pd.read_csv(TEXT_KEYWORD_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        keyword = self.data.loc[idx, 'keyword']
        return text, keyword