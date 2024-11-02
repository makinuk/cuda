import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer
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

# Model ve tokenizer'ı yükleyin
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_weights_path, config=config_path)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# API Sorgusu Yapısı
class Query(BaseModel):
    text: str

# Sorgu API'si
@app.post("/generate")
async def generate(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Dağıtık işlemi sonlandır
@app.on_event("shutdown")
def shutdown_event():
    torch.distributed.destroy_process_group()
