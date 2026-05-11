# Custom Neural Machine Translation (NMT) with Transformers

This repository contains a custom implementation of the Transformer architecture for Neural Machine Translation, translating from English to French. The Transformer architecture was build from scratch without using any libraries such as HuggingFace. The dataset was taken from https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench.


## Model Performance

I trained three variations of our custom Transformer model. Below are the hyperparameter configurations and their corresponding overall BLEU scores on the test set:

### 1. Large Model
- **Parameters**: `nhead` = 8, `d_model` = 512, `d_ff` = 2048, `num_layers` = 6
- **BLEU Score**: **58.99**

### 2. Large Model (with LR Warmup 4000)
- **Parameters**: `nhead` = 8, `d_model` = 512, `d_ff` = 2048, `num_layers` = 6
- **BLEU Score**: **51.67**
- *Note: Trained using a custom learning rate scheduler with a warmup phase of 4000 steps.*

### 3. Small Model
- **Parameters**: `nhead` = 4, `d_model` = 256, `d_ff` = 1024, `num_layers` = 3
- **BLEU Score**: **4.23**

---

## How to Run
Because the models I trained were too large to upload to github, this repo does not contain the model checkpoints.
### 1. Setup Environment
Install dependencies:
```bash
pip install scikit-learn torch tensorboard tqdm einops nltk
```

### 2. Preprocess Data
Convert raw text data (`.csv` format) into tokenized vocabulary mappings and PyTorch tensors:
```bash
cd src
python preprocess.py
```
*This will generate the vocabulary index files (`eng_vocab.pkl`, `fra_vocab.pkl`) and tokenized `.pt` tensor datasets in the `data/` directory.*

### 3. Train the Model
You can start training the custom Transformer model by running `train_Transformer.py`. By default, it runs with the **Large Model** configuration:
```bash
cd src
python train_Transformer.py
```

### 4. Translate Text
```bash
cd src
python translate_Transformer.py --model_path ../models/transformer_best_large.pt --sentence "take a seat ."
```
*Important: You must provide the corresponding architectural dimensions (like `--emb_size 512`) if you are loading a checkpoint that does not match the script's default fallback architecture.*

### 5. Evaluate BLEU Score
To run a full evaluation over the entire test set and output a comprehensive report (with the overall BLEU score and the first 10 translated sentences):
```bash
cd src
python compute_bleu_Transformer.py --model_path ../models/transformer_best_large.pt --beam_size 4 --patience 2.0 --emb_size 512 --num_layers 6 --ffn_hid_dim 2048
```
*The resulting report will be saved to the `run/` directory as `evaluation_report.txt`.*
