#!/bin/bash

# Script: gpu01_setup.sh
# Description: LLaMA 2 - 70B kurulumu için gerekli bağımlılıkların yüklenmesi, Python, CUDA ve DeepSpeed ayarları

# 1. Sistem Güncelleme
sudo apt update && sudo apt upgrade -y

# 2. Python Yükleme
sudo apt install -y python3 python3-pip

# Python'u sistem genelinde 'python' olarak tanımlayın
sudo ln -s /usr/bin/python3 /usr/bin/python
sudo ln -s /usr/bin/pip3 /usr/bin/pip

# 3. Gerekli Python Kütüphanelerinin Kurulumu
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install deepspeed transformers accelerate

# 4. LLaMA Modelini Yükleyin
mkdir -p /models/llama-2-70b
cd /models/llama-2-70b
# Model dosyalarını Hugging Face üzerinden indirin
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b', device_map='auto', offload_folder='offload')
"

# 6. DeepSpeed Yapılandırma Dosyası
cat <<EOT >> ds_config.json
{
  "train_batch_size": 2,
  "steps_per_print": 10,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
    "stage": 2
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5
    }
  },
  "fp16": {
    "enabled": true
  }
}
EOT

echo "gpu kurulumu tamamlandı."
