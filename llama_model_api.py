# llama_model_api.py
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# FastAPI başlat
app = FastAPI()

# Dağıtık yapılandırma
init_process_group(backend="nccl")

# GPU ID ve dünya boyutunu belirle
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "2"))  # İki makinada çalıştıracağımız için 2 olarak ayarlanır
device = torch.device(f"cuda:{local_rank}")

# Modeli yükleyin
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-70B-Instruct/")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Sorgu modelini tanımla
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
