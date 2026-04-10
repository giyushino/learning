# learning

![happy](https://media.tenor.com/Rukf_ikrukgAAAAe/happy-super-happy.png)

```bash
git clone git@github.com:giyushino/learning.git
cd learning
uv venv
source .venv/bin/activate
uv pip install -e .

# depending on CUDA version:
pip install --upgrade "jax[cuda12]"
pip install --upgrade "jax[cuda13]"
```


goals:
write triton kernels for transformer blocks, both forward and backward pass
get more familiar with jax
write a "decent" dataloader that does multithreading/multiprocessing (ie we load batches so we always keep GPUs hot)
train diffusion models
train flow matching models

