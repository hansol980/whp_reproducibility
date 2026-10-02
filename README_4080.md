# WHP reproduction on one RTX 4080

This fork keeps the upstream WHP/TOFU data flow but replaces the full-model,
multi-GPU fine-tune with a single-GPU 4-bit QLoRA run. It is a constrained
reproduction, not an exact reproduction of the original full fine-tuning setup.

## Hardware target

- Linux
- One RTX 4080 with 16 GB VRAM
- 100 GB or more of local storage

## Environment

```bash
conda create -n whp-repro python=3.10 -y
conda activate whp-repro
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install --upgrade pip packaging wheel
pip install -r requirements.txt
pip install flash-attn==2.5.3 --no-build-isolation
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
huggingface-cli login
```

Accept the model licenses before running:

- `meta-llama/Llama-2-7b-chat-hf`
- `locuslab/tofu_ft_llama2-7b`

## One-command smoke run

```bash
bash scripts/run_tofu_whp_4080.sh "$PWD/results" 0 forget01 1
```

The script creates WHP teacher targets, trains a 4-bit LoRA adapter, and evaluates
every saved adapter checkpoint. Teacher targets are cached and reused on later runs.

After the smoke run succeeds, increase the number of epochs:

```bash
bash scripts/run_tofu_whp_4080.sh "$PWD/results" 0 forget01 5
```

Results are written below `results/runs/`. The main summary for each checkpoint is
`checkpoint-*/eval/aggregate_stat.csv`.

## Reproduction scope

The run uses the follow-up repository's WHP baseline on TOFU. It does not contain
the Harry Potter novels and it does not reproduce the original paper's reinforced-
model stage. The 4-bit QLoRA adaptation also differs from the upstream full-model
fine-tuning configuration, so comparisons should use the same QLoRA setup for every
method.
