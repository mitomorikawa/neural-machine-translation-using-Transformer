import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import src.library.preprocessor as preprocessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_DataLoader():
    loader = preprocessor.DataLoader("../../data/eng_fra.csv")
    eng_texts, fr_texts = loader.load()
    assert eng_texts[0] == "Hi."
    assert fr_texts[0] == "Salut!"
    print("DataLoader test passed.")

def test_Standardizer():
    standardizer = preprocessor.Standardizer()
    texts = ["Hello World!", "  Étudiant!  ", "œéàèùçâêÎôûëïüç,,,,", "J'ai un chat.", "I'm a student."]
    standardized_texts = standardizer.standardize(texts)
    if standardized_texts != ["hello world !", "étudiant !" ,"œéàèùçâêîôûëïüç , , , ,", "j ' ai un chat .", "i ' m a student ."]:
        raise AssertionError(f"""Standardization did not produce the expected output.
                             Expected: ['hello world !', 'étudiant !', 'œéàèùçâêîôûëïüç , , , ,', 'j \' ai un chat .', 'i \' m a student .']
                                Got: {standardized_texts}
                             """)
    print("Standardization test passed.")

def test_Tokenizer():
    # The third src sentence is short but its tgt pair is too long —
    # both should be filtered out together to keep alignment.
    src_texts = [
        "hello world !",
        "j ' ai un chat .",
        "i ' m a student yeah .",    # 7 tokens — kept (src side)
    ]
    tgt_texts = [
        "bonjour monde !",
        "i have a cat .",
        "je suis vraiment un etudiant oui .", # 8 tokens — too long (tgt side)
    ]
    tokenizer = preprocessor.Tokenizer()
    src_tokenized, tgt_tokenized = tokenizer.word_tokenize(src_texts, tgt_texts, max_len=6)
    expected_src = [
        ["hello", "world", "!"],
        ["j", "'", "ai", "un", "chat", "."],
    ]
    expected_tgt = [
        ["bonjour", "monde", "!"],
        ["i", "have", "a", "cat", "."],
    ]
    if src_tokenized != expected_src:
        raise AssertionError(f"""Tokenization (src) did not produce the expected output.
                             Expected: {expected_src}
                                Got: {src_tokenized}
                             """)
    if tgt_tokenized != expected_tgt:
        raise AssertionError(f"""Tokenization (tgt) did not produce the expected output.
                             Expected: {expected_tgt}
                                Got: {tgt_tokenized}
                             """)
    if len(src_tokenized) != len(tgt_tokenized):
        raise AssertionError("src and tgt tokenized lists are not the same length (misaligned).")
    print("Tokenization test passed.")

def test_Indexer():
    tokenized_texts = [
        ["j", "'", "ai", "un", "chat", "."],
        ["i", "'", "m", "a","student","mate", "."]
    ]
    indexer = preprocessor.Indexer()
    indexer.build_vocab(tokenized_texts, min_freq=1)
    expected_word2idx = {"<sos>":0, "<eos>":1, "<pad>":2, "<unk>":3, 
                         "j":4, "'":5, "ai":6, "un":7, "chat":8, ".":9, 
                         "i":10, "m":11, "a":12, "student":13, "mate":14}
    expected_idx2word = {0:"<sos>", 1:"<eos>", 2:"<pad>", 3:"<unk>",
                            4:"j", 5:"'", 6:"ai", 7:"un", 8:"chat", 9:".", 
                            10:"i", 11:"m", 12:"a", 13:"student", 14:"mate"}
    expected_word2count = {"j":1, "'":2, "ai":1, "un":1, "chat":1, ".":2,
                           "i":1, "m":1, "a":1, "student":1, "mate":1}
    expected_vocab_size = 15
    expected_tensor_without_sos = [[4, 5, 6, 7, 8, 9,1,2], [10, 5, 11, 12, 13,14,9,1]]
    expected_tensor_with_sos = [[0, 4, 5, 6, 7, 8, 9,1,2], [0, 10, 5, 11, 12, 13,14,9,1]]
    tensor_without_sos = indexer.text_to_indices(tokenized_texts)
    tensor_with_sos = indexer.text_to_indices(tokenized_texts, prepend_sos=True)
    if indexer.word2idx != expected_word2idx:
        raise AssertionError(f"""Indexer did not produce the expected word2idx mapping.
                             Expected: {expected_word2idx}
                                Got: {indexer.word2idx}
                             """)
    if indexer.idx2word != expected_idx2word:
        raise AssertionError(f"""Indexer did not produce the expected idx2word mapping.
                             Expected: {expected_idx2word}
                                Got: {indexer.idx2word}
                             """)
    if indexer.word2count != expected_word2count:
        raise AssertionError(f"""Indexer did not produce the expected word2count mapping.
                             Expected: {expected_word2count}
                                Got: {indexer.word2count}
                             """)
    if indexer.vocab_size != expected_vocab_size:
        raise AssertionError(f"""Indexer did not produce the expected vocabulary size.
                             Expected: {expected_vocab_size}
                                Got: {indexer.vocab_size}
                             """)
    if tensor_without_sos!= expected_tensor_without_sos:
        raise AssertionError(f"""Indexer did not produce the expected tensor.
                             Expected: {expected_tensor_without_sos}
                                Got: {tensor_without_sos}
                             """)
    if tensor_with_sos!= expected_tensor_with_sos:
        raise AssertionError(f"""Indexer did not produce the expected tensor with <sos>.
                             Expected: {expected_tensor_with_sos}
                                Got: {tensor_with_sos}
                             """)
    
    
    print("Indexer test passed.")

def test_Splitter():
    splitter = preprocessor.Splitter(train_ratio=0.8, val_ratio=0.1)
    src_indices = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    tgt_indices = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    splitter.split(src_indices, tgt_indices)
    
    (src_train, src_val, src_test), (tgt_train, tgt_val, tgt_test) = splitter.split(src_indices, tgt_indices)
    print("src_train:", src_train)
    print("src_val:", src_val)
    print("src_test:", src_test)
    print("tgt_train:", tgt_train)
    print("tgt_val:", tgt_val)
    print("tgt_test:", tgt_test)
    if len(src_train) != 8 or len(src_val) != 1 or len(src_test) != 1 \
        or len(tgt_train) != 8 or len(tgt_val) != 1 or len(tgt_test) != 1:
        raise AssertionError("Splitter did not split the data correctly.")
        
    splitter.save_indices(src_train, "test_data/src_train.pt")
    splitter.save_indices(src_val, "test_data/src_val.pt")
    splitter.save_indices(src_test, "test_data/src_test.pt")
    splitter.save_indices(tgt_train, "test_data/tgt_train.pt")
    splitter.save_indices(tgt_val, "test_data/tgt_val.pt")
    splitter.save_indices(tgt_test, "test_data/tgt_test.pt")

    

    print("Splitter test passed.")

if __name__ == "__main__":
    test_DataLoader()
    test_Standardizer()
    test_Tokenizer()
    test_Indexer()
    test_Splitter()