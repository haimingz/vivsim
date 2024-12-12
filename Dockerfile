FROM python:3.10
RUN pip install -U "jax[cuda12]" tqdm matplotlib numpy

