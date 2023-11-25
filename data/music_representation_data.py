from torch.utils.data import Dataset
import pandas as pd

MUSIC_PRESENTATION_DATA_PATH = 'data/csv/music_representation_text.csv'
class MusicRepresentationData(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        keyword = self.data.loc[idx, 'representation']
        return text, keyword
