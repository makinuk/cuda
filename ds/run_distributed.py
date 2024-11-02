import os
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

def setup_model():
    model_path = "/root/.llama/checkpoints/Llama3.1-70B-Instruct/"
    
    # DeepSpeed config
    ds_config = "ds_config.json"
    dschf = HfDeepSpeedConfig(ds_config)

    # Model ve tokenizer yükleme
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )

    # DeepSpeed engine oluşturma
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    return model_engine, tokenizer

def generate_text(model_engine, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model_engine.device)
    
    outputs = model_engine.module.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Model ve tokenizer'ı yükle
    model_engine, tokenizer = setup_model()
    
    # Test et
    if torch.distributed.get_rank() == 0:  # Sadece ana node'da çıktı al
        prompt = "Merhaba, bugün hava nasıl?"
        response = generate_text(model_engine, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
