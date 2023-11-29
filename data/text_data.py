from torch.utils.data import Dataset
import pandas as pd

TEXT_DATA_PATH = 'data/csv/text_data.csv'

class RepresentationTextData(Dataset):
    def __init__(self):
        self.data = pd.read_csv(TEXT_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        
        return text
