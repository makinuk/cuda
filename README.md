# cuda





`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=<rank> --master_addr="<gpu01_ip>" --master_port=<port> llama_model_script.py`

- rank: gpu01 için 0, gpu02 için 1 olarak ayarlanmalıdır.
- <gpu01_ip> ve <port> bilgileri, ana sunucunun IP ve açık port bilgileri olmalıdır. (29400)

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="10.0.2.240" --master_port=29400 llama_model_script.py
