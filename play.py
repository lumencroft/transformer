import torch
from model import GPT, GPTConfig
import os

# -----------------------------------------------------------------------------
# 1. 설정 (학습 때랑 똑같이 맞춰야 해!)
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 1024
config = GPTConfig(
    vocab_size=256, 
    block_size=block_size,
    n_layer=12,      
    n_head=12,       
    n_embd=768,      
    dropout=0.0,     # 테스트할 땐 드롭아웃 꺼야 해
    use_conv=False
)

# -----------------------------------------------------------------------------
# 2. 모델 로드 (저장된 뇌 불러오기 🧠)
# -----------------------------------------------------------------------------
print(f"🔄 Loading model on {device}...")
model = GPT(config)

# 저장된 파일 없으면 혼난다?
if not os.path.exists('gpt_enwik8.pt'):
    raise FileNotFoundError("야! gpt_enwik8.pt 파일 어디 갔어? 학습 먼저 하고 와! ♡")

# state_dict 불러오기 (strict=True로 꼼꼼하게 체크)
model.load_state_dict(torch.load('gpt_enwik8.pt', map_location=device))
model.to(device)
model.eval() # 평가 모드 전환 (이거 안 하면 결과 이상하게 나와)
print("✅ Model loaded! Let's play.")

# -----------------------------------------------------------------------------
# 3. 무한 대화 루프
# -----------------------------------------------------------------------------
print("\n" + "="*40)
print(" 🤖 시콘의 AI 장난감 (Ctrl+C로 종료)")
print(" 영어로 아무 말이나 시작해 봐 (예: The history of)")
print("="*40 + "\n")

while True:
    try:
        start_text = input("You > ")
        if not start_text: continue

        # 1. 네가 쓴 글자를 숫자로 변환 (Encoding)
        context_ids = [ord(c) for c in start_text]
        x = torch.tensor([context_ids], dtype=torch.long, device=device)

        # 2. 모델이 뒷내용 상상하기 (Generating)
        # temperature: 높으면(1.0~) 창의적(아무말), 낮으면(0.8~) 보수적
        y = model.generate(x, max_new_tokens=200, temperature=0.8) 

        # 3. 숫자를 다시 글자로 변환 (Decoding)
        output_text = "".join([chr(i) for i in y[0].tolist()])
        
        print(f"AI  > {output_text[len(start_text):]}") # 네가 쓴 거 빼고 뒷부분만 출력
        print("-" * 40)

    except KeyboardInterrupt:
        print("\n\nBye! 재밌었어? ♡")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")