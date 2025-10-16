sky launch config.yaml -y --use-spot --infra=gcp -i 5 --down
ssh sky-466b-chhzh123
conda create -n sglang python=3.12
conda activate sglang
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
git checkout feat/grok
pip install --upgrade pip setuptools packaging
pip install -e "python[all]"