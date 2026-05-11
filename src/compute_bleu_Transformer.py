import torch
import os
import pickle
import argparse
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from library.nn_architectures import Seq2SeqTransformer
from translate_Transformer import beam_search, greedy_decode, load_vocab

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

def load_tensor(path):
    return torch.load(path, map_location=DEVICE, weights_only=True)

def indices_to_words(indices, idx2word):
    words = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx not in (SOS_IDX, PAD_IDX):
            words.append(idx2word.get(idx, "<unk>"))
    return " ".join(words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU score on test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing test data and vocab.")
    parser.add_argument("--output_dir", type=str, default="../run", help="Directory to save the evaluation report.")
    
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_hid_dim", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for beam search. Set to 1 for greedy decoding.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Length penalty.")
    parser.add_argument("--patience", type=float, default=2.0, help="Patience factor.")
    parser.add_argument("--max_len", type=int, default=64, help="Maximum length of generated sequence.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    
    print("Loading vocabs...")
    eng_vocab_full = load_vocab(os.path.join(args.data_dir, "eng_vocab.pkl"))
    fra_vocab_full = load_vocab(os.path.join(args.data_dir, "fra_vocab.pkl"))
    
    src_vocab_size = eng_vocab_full["vocab_size"]
    tgt_vocab_size = fra_vocab_full["vocab_size"]
    src_idx2word = eng_vocab_full["idx2word"]
    tgt_idx2word = fra_vocab_full["idx2word"]
    
    print("Loading test data...")
    src_test = load_tensor(os.path.join(args.data_dir, "eng_test.pt"))
    tgt_test = load_tensor(os.path.join(args.data_dir, "fra_test.pt"))
    
    print(f"src_test shape: {src_test.shape}, tgt_test shape: {tgt_test.shape}")
    
    print("Building model...")
    transformer = Seq2SeqTransformer(
        d_model=args.emb_size,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.ffn_hid_dim,
        seq_len=args.seq_len,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout=args.dropout,
    ).to(DEVICE)
    
    print(f"Loading weights from {args.model_path}...")
    transformer.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
    transformer.eval()
    
    references = []
    hypotheses = []
    
    sample_translations = []
    
    print("Translating test set...")
    for i in tqdm(range(len(src_test)), desc="Translating"):
        src_tensor = src_test[i].unsqueeze(0).to(DEVICE) # (1, S)
        tgt_tensor = tgt_test[i]
        
        # Ground truth
        ref_words = indices_to_words(tgt_tensor.tolist(), tgt_idx2word)
        references.append([ref_words.split()])
        
        # Source
        src_words = indices_to_words(src_tensor[0].tolist(), src_idx2word)
        
        # Decode
        if args.beam_size > 1:
            pred_tensor = beam_search(
                transformer, 
                src_tensor, 
                max_len=args.max_len, 
                start_symbol=SOS_IDX, 
                beam_size=args.beam_size, 
                alpha=args.alpha, 
                patience=args.patience
            ).flatten()
        else:
            pred_tensor = greedy_decode(
                transformer,
                src_tensor,
                max_len=args.max_len,
                start_symbol=SOS_IDX
            ).flatten()
            
        pred_words = indices_to_words(pred_tensor.tolist(), tgt_idx2word)
        hypotheses.append(pred_words.split())
        
        if i < 10:
            sample_translations.append({
                "source": src_words,
                "reference": ref_words,
                "prediction": pred_words
            })
            
    print("Computing BLEU score...")
    # corpus_bleu expects a list of list of reference words, and a list of hypothesis words
    bleu_score = corpus_bleu(references, hypotheses) * 100 # Convert to percentage
    
    print(f"Overall BLEU score: {bleu_score:.2f}")
    
    print(f"Saving report to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== NMT Evaluation Report ===\n")
        f.write(f"Model Checkpoint: {args.model_path}\n")
        f.write(f"Beam Size: {args.beam_size}, Alpha: {args.alpha}, Patience: {args.patience}\n")
        f.write(f"Overall BLEU Score: {bleu_score:.2f}\n\n")
        
        f.write("=== First 10 Sample Translations ===\n")
        for idx, sample in enumerate(sample_translations):
            f.write(f"Sample {idx+1}:\n")
            f.write(f"Source:     {sample['source']}\n")
            f.write(f"Reference:  {sample['reference']}\n")
            f.write(f"Prediction: {sample['prediction']}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()
