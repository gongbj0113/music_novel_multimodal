from torch.utils.data import Dataset
import pandas as pd

REPRESENTATION_TEXT_DATA_PATH = 'data/csv/music_representation_text.csv'

class RepresentationTextData(Dataset):
    def __init__(self):
        self.data = pd.read_csv(REPRESENTATION_TEXT_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        representation = self.data.iloc[idx]['representation']
        text = self.data.iloc[idx]['text']
        return representation, text