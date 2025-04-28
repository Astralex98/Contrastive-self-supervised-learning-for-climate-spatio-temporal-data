# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /prj
COPY . .
RUN chmod +x install.sh; ./install.sh
# CMD ["earth_former/bin/python", "scripts/cuboid_transformer/pdsi/train_cuboid_pdsi.py", "--cfg cfg.yaml", "--ckpt_name last.ckpt", "--save tmp_pdsi"]
CMD ["ts2vec/bin/jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
