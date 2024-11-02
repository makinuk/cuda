import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Model dosyasının ve tokenizer'ın yollarını belirleyin
model_path = "/root/.llama/checkpoints/Llama3.1-70B-Instruct/"
model_weights_path = model_path + "consolidated.00.pth"  # veya indirdiğiniz model dosyasının adı
config_path = model_path + "config.json"
tokenizer_path = model_path + "tokenizer.model"  # tokenizer dosyanızın adı

# Tokenizer'ı yükleyin
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Modeli yükleyin
model = LlamaForCausalLM.from_pretrained(model_weights_path, config=config_path)
model.eval()  # İnference için değerlendirme moduna geçin

# Modelin çalışıp çalışmadığını test edin
input_text = "Merhaba dünya!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Çıktı:", output_text)
