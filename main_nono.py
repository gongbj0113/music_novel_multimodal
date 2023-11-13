import torch
from transformers import BertModel, TrainingArguments, Trainer
from datasets import load_dataset
from kobert_tokenizer import KoBERTTokenizer

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Available: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU Unavailable")

# 1. 데이터셋 생성 (예시 데이터셋 사용)
dataset = load_dataset("glue", "mrpc")

# 2. 토크나이저 및 모델 로드 (skt/kobert 사용)
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, labels=2)  # 분류 클래스 수에 맞게 설정

# 모델과 데이터를 GPU로 이동
model.to(device)

# 3. 학습 설정
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    eval_steps=100,  # 평가 주기 설정
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    save_total_limit=5,
    save_steps=500,  # 모델 저장 주기 설정
    learning_rate=2e-5,
    logging_steps=100,  # 로깅 주기 설정
    do_train=True,
    do_eval=True,
)


# 4. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

# 5. Fine-tuning 실행
trainer.train()

# 6. 모델 저장
model.save_pretrained("./fine-tuned-kobert")
tokenizer.save_pretrained("./fine-tuned-kobert")

# 7. 평가
results = trainer.evaluate()
print(results)