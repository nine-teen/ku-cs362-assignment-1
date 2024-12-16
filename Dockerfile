FROM python:3.13-bookworm

RUN apt-get update \
    && apt-get install -y vim gnupg2 bash-completion git \
    && echo "source /usr/share/bash-completion/bash_completion" >> ~/.bashrc

WORKDIR /workspaces

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
