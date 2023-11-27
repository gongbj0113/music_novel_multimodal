from torch.utils.data import Dataset
import pandas as pd

MUSIC_PRESENTATION_DATA_PATH = 'data/csv/music_representation_text.csv'
class MusicRepresentationData(Dataset):
    def __init__(self):
        self.data = pd.read_csv(MUSIC_PRESENTATION_DATA_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        music = self.data.iloc[idx]['music']
        representation = self.data.iloc[idx]['representation']
        return music, representation
