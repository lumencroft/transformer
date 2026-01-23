import torch
import numpy as np
from model import GPT, GPTConfig
import math
import time
import os
import urllib.request
import zipfile

# -----------------------------------------------------------------------------
# 1. 설정 (Hyperparameters) - 실험용 Nano 모드 ⚡
# -----------------------------------------------------------------------------
batch_size = 64        # 4090이니까 배치는 넉넉하게
block_size = 256       # 1024는 너무 깁니다. 실험용으론 256이면 패턴 보기에 충분합니다.
max_iters = 5000       # 스텝 수는 유지 (학습 곡선 비교를 위해)
eval_interval = 250    # 평가는 자주
learning_rate = 1e-3   # 모델이 작으니까 학습률은 다시 높입니다
device = 'cuda'

# -----------------------------------------------------------------------------
# GPU 진짜 쓰는지 확인하는 코드 (너의 의심병 치유용 ♡)
# -----------------------------------------------------------------------------
if device == 'cuda':
    print(f"🔥 GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Usage Check: {torch.cuda.memory_allocated() / 1024**2:.2f} MB used")
else:
    print("⚠️ Warning: CPU is running... Something is wrong!")

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
# 3. 모델 초기화 (NanoGPT 체급)
# -----------------------------------------------------------------------------
config = GPTConfig(
    vocab_size=256, 
    block_size=block_size,
    n_layer=6,       # 12 -> 6 (층수 절반)
    n_head=6,        # 12 -> 6 (헤드 절반)
    n_embd=384,      # 768 -> 384 (임베딩 절반)
    dropout=0.2,     # 작은 모델은 과적합 될 수 있으니 드롭아웃 좀 더 줌
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


# -----------------------------------------------------------------------------
# 6. 모델 저장 (날려먹지 말자 제발 ♡)
# -----------------------------------------------------------------------------
print("\n💾 Saving model...")
torch.save(model.state_dict(), 'gpt_enwik8.pt')
print("✅ Model saved to 'gpt_enwik8.pt'")

# -----------------------------------------------------------------------------
# 7. 생성 테스트 (Inference) - 얘가 뭘 배웠나 보자!
# -----------------------------------------------------------------------------
print("\n📝 Generating text...")
model.eval()

# 시작 문맥 (Context): "The" 라는 단어로 시작해볼게
context = torch.tensor([[ord('T'), ord('h'), ord('e')]], dtype=torch.long, device=device)

# 500글자 생성해줘!
generated = model.generate(context, max_new_tokens=500)

# 바이트를 문자로 디코딩 (깨진 문자는 무시)
output_text = "".join([chr(i) for i in generated[0].tolist()])
print("=" * 50)
print(output_text)
print("=" * 50)