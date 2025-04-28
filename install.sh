#!/usr/bin/env bash

apt-get update -y
pip install virtualenv
virtualenv ts2vec
. ts2vec/bin/activate
# pip install --upgrade setuptools pip
pip install -r requirements.txt
