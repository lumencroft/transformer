import torch
import numpy as np
from model import GPT, GPTConfig
import math
import time
import os
import urllib.request
import zipfile

# -----------------------------------------------------------------------------
# 1. 설정 (Hyperparameters) - 니가 나중에 바꿔야 할 수도 있어
# -----------------------------------------------------------------------------
batch_size = 32        # 64 -> 32 (메모리 부담 줄이기)
block_size = 128       # 256 -> 128 (Attention 연산량 4배 감소 효과)
max_iters = 5000  # 테스트용으로 짧게 잡음. 실제론 더 늘려야 해
eval_interval = 500
learning_rate = 1e-3   # 모델 작아지니까 학습률 좀 높이자
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): device = 'mps' # 맥북 쓰는 거 아니지? 혹시 몰라 넣음

# -----------------------------------------------------------------------------
# 2. 데이터 준비 (Data Prep) - "야생의" 방식으로 직접 로드
# -----------------------------------------------------------------------------
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

file_path = os.path.join(data_dir, 'enwik8')
zip_path = os.path.join(data_dir, 'enwik8.zip')

# 파일이 없으면 다운로드 (Matt Mahoney의 원본 사이트)
if not os.path.exists(file_path):
    print("📥 Downloading enwik8 from source...")
    url = "http://mattmahoney.net/dc/enwik8.zip"
    urllib.request.urlretrieve(url, zip_path)
    print("📦 Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
else:
    print("✅ enwik8 already exists. Skipping download.")

# 파일 읽기 (rb 모드 = Raw Bytes)
print("📂 Reading raw bytes...")
with open(file_path, 'rb') as f:
    raw_data = f.read() # bytes 타입으로 읽힘

# 바이트(0~255)를 정수 텐서로 변환
# numpy를 거쳐서 tensor로 만드는 게 속도가 빨라
print("🔄 Converting to tensor...")
data_tensor = torch.from_numpy(np.frombuffer(raw_data, dtype=np.uint8).copy()).long()

n = len(data_tensor)
# 문제 조건: 90M Train / 5M Dev / 5M Test
train_data = data_tensor[:90_000_000]
val_data = data_tensor[90_000_000:95_000_000]
test_data = data_tensor[95_000_000:]

print(f"Dataset Split Completed! Total bytes: {n}")
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# -----------------------------------------------------------------------------
# 3. 모델 초기화 (Model Init)
# -----------------------------------------------------------------------------
# enwik8은 byte 단위니까 vocab_size는 무조건 256이야.
config = GPTConfig(
    vocab_size=256, 
    block_size=block_size,
    n_layer=4,      # 베이스라인이니까 가볍게 시작
    n_head=4, 
    n_embd=128,
    dropout=0.0,
    use_conv=False
)
model = GPT(config)
model.to(device)
print(f"🤖 Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------
# 4. 유틸리티 함수 (Utils)
# -----------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# train.py 함수 수정
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20)  # <--- 원래 200이었음. 20으로 줄여!
        for k in range(20):       # <--- 여기도 20으로!
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# 5. 학습 루프 (Training Loop) - 수다쟁이 모드
# -----------------------------------------------------------------------------
print("🚀 Training Start!")
start_time = time.time()

for iter in range(max_iters):
    # 주기적으로 평가 (오래 걸림)
    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        losses = estimate_loss()
        train_bpc = losses['train'] / math.log(2)
        val_bpc = losses['val'] / math.log(2)
        print(f"\n[Step {iter}] time: {elapsed:.2f}s | train: {train_bpc:.4f} bpc | val: {val_bpc:.4f} bpc")

    # 학습 진행 (여기서 멈춘 것처럼 보였던 거임)
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # ★★★ 생존 신고 추가 ★★★
    # 줄바꿈 없이 점(.)만 찍어서 진행 상황 보여줌
    print(".", end="", flush=True) 

print("\n🏁 Training Finished!")