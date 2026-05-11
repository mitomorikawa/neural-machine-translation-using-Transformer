import torch
import math
from torch import Tensor
import pickle
import argparse
import os

# Import the custom Seq2SeqTransformer and create_masks
from library.nn_architectures import Seq2SeqTransformer, create_masks
from library.preprocessor import Tokenizer, Standardizer

# Must match training script
SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path, model):
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────────
# Main translate function
# ─────────────────────────────────────────────────────────────
def translate(model, sentence, src_vocab, tgt_vocab, tokenizer, max_len=50, beam_size=1, alpha=0.7, patience=1.0):
    """
    model       : trained Seq2SeqTransformer
    sentence    : raw input string (English)
    src_vocab   : dict {token -> index}
    tgt_vocab   : dict {index -> token}
    tokenizer   : function that splits sentence into tokens
    """

    model.eval()

    # ── 1. Tokenize & numericalize ────────────────────────────
    tokens = tokenizer(sentence.lower())
    tokens = tokens + ["<eos>"]

    src_indexes = [
        src_vocab.get(tok, UNK_IDX) for tok in tokens
    ]

    # Custom model expects batch_first: (batch_size, seq_len)
    src = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)  # (1, S)

    # ── 2. Decode ─────────────────────────────────────────────
    if beam_size > 1:
        tgt_tokens = beam_search(
            model,
            src,
            max_len=max_len,
            start_symbol=SOS_IDX,
            beam_size=beam_size,
            alpha=alpha,
            patience=patience
        ).flatten()
    else:
        tgt_tokens = greedy_decode(
            model,
            src,
            max_len=max_len,
            start_symbol=SOS_IDX
        ).flatten()

    # ── 3. Convert indices → tokens ───────────────────────────
    tgt_tokens = tgt_tokens.cpu().numpy()

    words = []
    for idx in tgt_tokens:
        if idx == EOS_IDX:
            break
        if idx not in (SOS_IDX, PAD_IDX):
            words.append(tgt_vocab.get(idx, "<unk>"))

    return " ".join(words)

# ─────────────────────────────────────────────────────────────
# Beam Search decode for custom Seq2SeqTransformer
# ─────────────────────────────────────────────────────────────
def beam_search(model, src, max_len, start_symbol, beam_size=4, alpha=0.7, patience=1.0):
    src = src.to(DEVICE)
    # create dummy target for src mask (since tgt is empty)
    src_padding_mask, _, _ = create_masks(src, src, src.size(1))
    
    # Encode
    src_embed = model.src_embedding(src) * math.sqrt(model.d_model)
    src_embed = model.positional_encoding(src_embed)
    memory = src_embed
    for layer in model.encoder_layers:
        memory = layer(memory, src_padding_mask)
        
    # Expand memory and src_padding_mask to process all beams in parallel
    memory = memory.repeat(beam_size, 1, 1)
    src_padding_mask = src_padding_mask.repeat(beam_size, 1, 1, 1)

    # Initialize beams: [(score, sequence)]
    active_beams = [(0.0, [start_symbol])]
    completed_beams = []

    for step in range(max_len - 1):
        current_k = len(active_beams)
        
        # Prepare input tensor for current active beams
        ys_list = [beam[1] for beam in active_beams]
        ys = torch.tensor(ys_list, dtype=torch.long, device=DEVICE) # (current_k, seq_len)
        
        # We only need memory for current_k
        current_memory = memory[:current_k]
        current_src_padding_mask = src_padding_mask[:current_k]
        
        # Create masks
        _, tgt_padding_mask, causal_mask = create_masks(src.repeat(current_k, 1), ys, ys.size(1))
        
        # Decode
        tgt_embed = model.tgt_embedding(ys) * math.sqrt(model.d_model)
        tgt_embed = model.positional_encoding(tgt_embed)
        dec_out = tgt_embed
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, current_memory, current_src_padding_mask, tgt_padding_mask, causal_mask)
            
        # Get probabilities for the last token
        logits = model.output_linear(dec_out[:, -1]) # (current_k, vocab_size)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (current_k, vocab_size)
        log_probs[:, PAD_IDX] = -float('inf') #ban <pad>
        log_probs[:, SOS_IDX] = -float('inf') #ban <sos>
        # Get top-k next tokens for each active beam
        topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)
        
        new_beams = []
        for i, (score, seq) in enumerate(active_beams):
            for j in range(beam_size):
                next_word = topk_indices[i, j].item()
                next_score = score + topk_log_probs[i, j].item()
                new_len = len(seq) + 1
                length_penalty = ((5 + new_len) / 6) ** alpha
                normalized_score = next_score / length_penalty
                new_beams.append((normalized_score, seq + [next_word]))
                
        # Sort new beams by score and keep top-k
        new_beams.sort(key=lambda x: x[0], reverse=True)
        new_beams = new_beams[:beam_size]
        
        # Check for completion
        active_beams = []
        for score, seq in new_beams:
            if seq[-1] == EOS_IDX:
                completed_beams.append((score, seq))
            else:
                score *= ((5 + len(seq)) / 6) ** alpha
                active_beams.append((score, seq))
                
        # Stop early if we have enough completed beams (using patience factor)
        required_completed = max(1, int(beam_size * patience))
        if len(completed_beams) >= required_completed:
            break
            
        if len(active_beams) == 0:
            break

    # If no completed beams, use the best active beam
    if len(completed_beams) == 0:
        active_beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = active_beams[0][1]
    else:
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed_beams[0][1]
        
    return torch.tensor([best_seq], dtype=torch.long, device=DEVICE)


# ─────────────────────────────────────────────────────────────
# Greedy decode for custom Seq2SeqTransformer
# ─────────────────────────────────────────────────────────────
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    # create dummy target for src mask (since tgt is empty)
    src_padding_mask, _, _ = create_masks(src, src, src.size(1))
    
    # Encode
    src_embed = model.src_embedding(src) * math.sqrt(model.d_model)
    src_embed = model.positional_encoding(src_embed)
    memory = src_embed
    for layer in model.encoder_layers:
        memory = layer(memory, src_padding_mask)
    
    # Decoder starts with <sos>
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    for _ in range(max_len - 1):
        _, tgt_padding_mask, causal_mask = create_masks(src, ys, ys.size(1))
        
        # Decode
        tgt_embed = model.tgt_embedding(ys) * math.sqrt(model.d_model)
        tgt_embed = model.positional_encoding(tgt_embed)
        dec_out = tgt_embed
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, memory, src_padding_mask, tgt_padding_mask, causal_mask)
            
        prob = model.output_linear(dec_out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if next_word == EOS_IDX:
            break
    return ys


def main():
    parser = argparse.ArgumentParser(description="Translate using trained custom Seq2SeqTransformer")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data")

    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_hid_dim", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for beam search. Set to 1 for greedy decoding.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Length penalty for beam search.")
    parser.add_argument("--patience", type=float, default=2.0, help="Patience factor for stopping criterion. (e.g. 0.5 or 2.0)")

    args = parser.parse_args()

    # ── Load vocabs ───────────────────────────────────────────
    eng_vocab_full = load_vocab(os.path.join(args.data_dir, "eng_vocab.pkl"))
    fra_vocab_full = load_vocab(os.path.join(args.data_dir, "fra_vocab.pkl"))

    src_vocab = eng_vocab_full["word2idx"]
    tgt_vocab = fra_vocab_full["idx2word"]

    src_vocab_size = eng_vocab_full["vocab_size"]
    tgt_vocab_size = fra_vocab_full["vocab_size"]

    # ── Build model ───────────────────────────────────────────
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

    # ── Load weights ──────────────────────────────────────────
    transformer = load_model(args.model_path, transformer)

    standardizer = Standardizer()

    def tokenizer(sentence: str):
        sentence = standardizer.standardize([sentence])[0]
        tokens = sentence.split()  
        return tokens

    translation = translate(
        transformer,
        args.sentence,
        src_vocab,
        tgt_vocab,
        tokenizer,
        beam_size=args.beam_size,
        alpha=args.alpha,
        patience=args.patience
    )

    print(f"Input: {args.sentence}")
    print(f"Output: {translation}")

if __name__ == "__main__":
    main()
