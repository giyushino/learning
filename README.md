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
