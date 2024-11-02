import os
import torch
import sentencepiece as spm
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlamaForCausalLM
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# FastAPI uygulaması başlat
app = FastAPI()

# Dağıtık yapılandırmayı başlat
init_process_group(backend="nccl")

# GPU ID ve dünya boyutunu belirle
local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")

# Model ve Tokenizer Yolları
model_path = "/root/.llama/checkpoints/Llama3.1-70B-Instruct/"
model_weights_path = os.path.join(model_path, "consolidated.00.pth")
config_path = os.path.join(model_path, "config.json")
tokenizer_model_path = os.path.join(model_path, "tokenizer.model")

# SentencePiece Tokenizer'ı Yükleyin
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)

# Modeli yükleyin
model = LlamaForCausalLM.from_pretrained(model_weights_path, config=config_path)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# API Sorgusu Yapısı
class Query(BaseModel):
    text: str

# Sorgu API'si
@app.post("/generate")
async def generate(query: Query):
    input_ids = torch.tensor([sp.encode(query.text)], device=device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=100)
    response = sp.decode(outputs[0].tolist())
    return {"response": response}

# Dağıtık işlemi sonlandır
@app.on_event("shutdown")
def shutdown_event():
    torch.distributed.destroy_process_group()
