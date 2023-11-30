# https://github.com/gyunggyung/KoGPT2-FineTuning/blob/master/main.py

import torch
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer 
from tqdm import tqdm
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook
	

# 라이브러리 임포트
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

# 데이터셋 클래스 정의
class KogptDataset(Dataset):
    def __init__(self, representation_text_dataset, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 데이터 로드 및 전처리
        lines = [
            data[0] + " > " + data[1] for data in representation_text_dataset
        ] # representation + text
        
        self.datas = tokenizer.batch_encode_plus(lines, add_special_tokens=True, padding="max_length", max_length=block_size, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        return torch.tensor(self.datas[i], dtype=torch.long)

class TextGeneratorFromRepresentation:
    def __init__(self):
        # 모델 학습 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
        self.loaded = False

    def load(self):
        model_path = "model/save/text_generator"
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.model.to(self.device)
        self.loaded = True

    def train(self, representation_text_data):
        # 데이터셋 및 DataLoader 준비
        dataset = KogptDataset(representation_text_data, self.tokenizer, block_size=128)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        num_epochs = 200
        warmup_ratio = 0.1
        log_interval = 200
        max_grad_norm = 1
        
        t_total = len(data_loader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        self.model.to(self.device)

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            for batch_id, label in enumerate(tqdm_notebook(data_loader)):
                optimizer.zero_grad()
                inputs = label.to(self.device)
                out = self.model(inputs, labels=inputs)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                if batch_id % log_interval == 0:
                    print("epoch {} batch id {} loss {}".format(epoch + 1, batch_id + 1, loss.data.cpu().numpy()), flush=True)

        # 모델 저장
        # If model/save directory does not exist, create it
        import os
        if not os.path.exists('model/save'):
            os.makedirs('model/save')
        # Save the model

        model_path = "model/save/text_generator"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        self.loaded = True
        
    def predict(self, representation):
        if not self.loaded:
            self.load()

        input_ids = self.tokenizer.encode(representation + " > ", return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=128, repetition_penalty=2.0,
                           pad_token_id=self.tokenizer.pad_token_id,
                           eos_token_id=self.tokenizer.eos_token_id,
                           bos_token_id=self.tokenizer.bos_token_id,
                           use_cache=True)
        
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # remove representation
        result = result.replace(representation + " > ", "").strip()
        return result