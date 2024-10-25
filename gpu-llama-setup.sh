#!/bin/bash

pip install -U "huggingface_hub[cli]" transformers
huggingface-cli login
#Enter access token
python llama-test.py