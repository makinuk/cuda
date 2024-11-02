# cuda





`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=<rank> --master_addr="<gpu01_ip>" --master_port=<port> llama_model_script.py`

- rank: gpu01 için 0, gpu02 için 1 olarak ayarlanmalıdır.
- <gpu01_ip> ve <port> bilgileri, ana sunucunun IP ve açık port bilgileri olmalıdır. (29400)

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.0.2.240" --master_port=29400 llama_model_script.py




python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=<rank> --master_addr="<gpu01_ip>" --master_port=29500 llama_model_api.py

- --nproc_per_node=2: Her makinede iki GPU bulunduğundan her biri için bir süreç oluşturuyoruz.
- --nnodes=2: Toplamda iki makine (node) bulunduğunu belirtiyoruz.
- --node_rank: gpu01 için 0, gpu02 için 1 olarak ayarlanmalıdır.
- --master_addr: gpu01 IP adresi.
- --master_port: Kullanılmayan bir port, örneğin 29500.


- GPU01 : `python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.0.2.240" --master_port=29500 llama_model_api.py`
- GPU02 : `python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="10.0.2.240" --master_port=29500 llama_model_api.py`
