from data.music_representation_data import MusicRepresentationData
import torch
import os
from model.music_captioning.model.bart import BartCaptionModel
from model.music_captioning.utils.audio_utils import load_audio, STR_CH_FIRST
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook

def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

class BartDataset(Dataset):
    def __init__(self, music_representation_data:MusicRepresentationData):
        self.datas = music_representation_data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        music, representation = self.datas[i]
        audio_tensor = get_audio(audio_path = music)
        audio_tensor = audio_tensor[0,:]
        # print("audio_tensor", audio_tensor.shape)
        return audio_tensor, representation

class RepresentationGeneratorFromMusic:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartCaptionModel(max_length = 128)

        if os.path.isfile("model/save/music_cap_transfer.pth") == False:
            torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth', 'model/save/music_cap_transfer.pth')

        pretrained_object = torch.load('model/save/music_cap_transfer.pth', map_location='cpu')
        state_dict = pretrained_object['state_dict']
        self.model.load_state_dict(state_dict)

        self.loaded = False
    
    def load(self):
        model_path = "model/save/music_cap_trained.pth"
        self.model.load_state_dict(torch.load(model_path))

        self.loaded = True


    def train(self, music_representation_data:MusicRepresentationData):
        # if torch.cuda.is_available():
        #     torch.cuda.set_device(self.device)
        #     self.model = self.model.cuda(self.device)
    
        # 데이터셋 및 DataLoader 준비
        dataset = BartDataset(music_representation_data)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        num_epochs = 30
        warmup_ratio = 0.1
        log_interval = 200
        max_grad_norm = 1
        
        t_total = len(data_loader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
        
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            for batch_id, (audio_tensor, representation) in enumerate(tqdm_notebook(data_loader)):
                optimizer.zero_grad()
                audio_tensor = audio_tensor.to(self.device)
                loss = self.model(audio_tensor, representation)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                if batch_id % log_interval == 0:
                    print("epoch {} batch id {} loss {}".format(epoch + 1, batch_id + 1, loss.data.cpu().numpy()), flush=True)
        
        # save model
        # If model/save directory does not exist, create it
        import os
        if not os.path.exists('model/save'):
            os.makedirs('model/save')
        # Save the model
        model_path = "model/save/music_cap_trained.pth"
        torch.save(self.model.state_dict(), model_path)

    def predict(self, music_path):
        if not self.loaded:
            self.load()
        
        self.model.to(self.device)
        audio_tensor = get_audio(audio_path = music_path)
        if self.device is not None:
            audio_tensor = audio_tensor[0,:].unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                samples=audio_tensor,
                num_beams=5,
            )
        
        return output[0]