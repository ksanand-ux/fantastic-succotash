# fantastic-succotash

This repository now includes `micro_llm.py`, a minimal educational GPT-style character language model in Python using PyTorch.

## Quick start

```bash
python micro_llm.py --text "hello world hello world hello world" --steps 200 --generate "hello " --tokens 80
```

Or train from a text file:

```bash
python micro_llm.py --file ./my_corpus.txt --steps 1000 --generate "Once upon " --tokens 200
```
