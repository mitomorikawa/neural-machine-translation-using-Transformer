""" 
This module contains classes for loading, standardizing, tokenizing, and indexing text data.
"""

import csv
import pickle
import unicodedata
import re
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataLoader:
    """ 
    This class loads raw text data from files and stores them in lists.
        attributes:
            - file_paths (list): List of file paths to load data from.
    
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load(self, header=1):
        """
        Loads data from the specified file paths and returns two list of strings.
        Params:
            header (int): The number of header rows to skip in the CSV file.
        returns: 
            Tuple(List[str], List[str]) - A tuple containing two lists of strings, one for each file. 
            The first is English and the second is French.
                        
        """
        eng_texts = []
        fra_texts = []

        with open(self.file_paths, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for _ in range(header):
                next(reader)  # Skip header row
            rows = list(reader)
            for row in tqdm(rows, desc="Loading data"):
                if len(row) == 2:
                    eng_texts.append(row[0])
                    fra_texts.append(row[1])
                else:
                    continue
        return eng_texts, fra_texts
    
class Standardizer:
    """ 
    This class standardizes text by converting it to lowercase and removing extra spaces.
    """
    def standardize(self, texts):
        """
        Takes a list of strings and returns a standardized version of them,
        preserving punctuation as separate tokens.
        
        Parameters:
            texts (List[str]): List of texts to standardize.
        
        Returns:
            List[str]: A list of standardized strings with token-friendly formatting.
        """
        standardized_texts = []
        for text in tqdm(texts, desc="Standardizing texts"):
            # Convert to lowercase and strip leading/trailing spaces
            standardized_text = text.lower().strip()
            
            # Add spaces around punctuation (keep Unicode letters and numbers untouched)
            standardized_text = re.sub(r'([^\w\sÀ-ÿ])', r' \1 ', standardized_text)

            # Remove anything that's not a letter, number, space, apostrophe, or punctuation
            standardized_text = ''.join(
                char for char in standardized_text
                if (
                    unicodedata.category(char).startswith(('L', 'N'))
                    or char.isspace()
                    or char in ".,!?;:'\"-()[]{}"
                )
            )

            # Remove extra spaces
            standardized_text = ' '.join(standardized_text.split())
            standardized_texts.append(standardized_text)

        return standardized_texts
    
class Tokenizer:
    """ 
    This class tokenizes text into words and removes punctuation.
    """
    def word_tokenize(self, src_texts, tgt_texts, max_len):
        """
        Tokenizes src and tgt texts together, keeping only pairs where BOTH
        sides are within max_len words. This prevents index misalignment
        between the two languages.

        Params:
            src_texts (List[str]): Source language texts.
            tgt_texts (List[str]): Target language texts.
            max_len (int): Maximum number of tokens allowed per sentence.
        Returns:
            Tuple[List[List[str]], List[List[str]]]: Aligned tokenized pairs.
        """
        src_tokenized, tgt_tokenized = [], []
        for src, tgt in tqdm(zip(src_texts, tgt_texts), total=len(src_texts)):
            src_tokens = src.split()
            tgt_tokens = tgt.split()
            if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                src_tokenized.append(src_tokens)
                tgt_tokenized.append(tgt_tokens)
        return src_tokenized, tgt_tokenized
    
class Indexer:
    """ 
    This class creates a vocabulary from tokens and converts them into indices based on a vocabulary.
        attributes:
            - dict word2idx: A dictionary mapping words to their indices.
            - dict idx2word: A dictionary mapping indices to their words.
            - dict word2count: A dictionary mapping words to their counts.
            - int vocab_size: The size of the vocabulary.
    """
    def __init__(self):
        self.word2idx = {"<sos>": 0, "<eos>": 1, "<pad>": 2, "<unk>":3}  # Start with special tokens
        self.idx2word = {0: "<sos>", 1: "<eos>", 2: "<pad>", 3:"<unk>"}  # Reverse mapping for special tokens
        self.word2count = {}
        self.vocab_size = 4
        
    def build_vocab(self, tokens, min_freq=2):
        """ 
        Builds a vocabulary from a list of tokens.
                params:
                    - List[List[str]] tokens: List of lists of tokens.
                returns:
                    None - The method updates the word2idx and idx2word dictionaries.
        """
        for text in tqdm(tokens, desc="Counting words"):
            for word in text:
                self.word2count[word] = self.word2count.get(word, 0) + 1

        # Now add only words with count >= min_freq
        for word, count in tqdm(self.word2count.items(), desc="Building vocabulary"):
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def text_to_indices(self, texts, prepend_sos=False, verbose=True):
        """ 
        Converts a list of texts into indices based on the vocabulary, padsding to a maximum sentence length and replaces uncommon words with <unk>.
                params:
                    - List[List[str]] texts: List of texts to convert.
                returns:
                    List[List[int]] - A list of lists of indices corresponding to the input texts.
        """
        max_length = max(len(text) for text in texts)
        if prepend_sos:
            indices = [[self.word2idx["<sos>"]]+[2 for _ in range(max_length)] for _ in range(len(texts))]  
            for i, text in enumerate(tqdm(texts)):
                for j, word in enumerate(text):
                    if word in self.word2idx:
                        indices[i][j+1] = self.word2idx[word]
                    else:
                        indices[i][j+1] = self.word2idx["<unk>"]
                indices[i].insert(len(text) + 1, self.word2idx["<eos>"])  # Set <eos> at the end of the sentence
            if verbose:
                print("max_length:", max_length+2)
        else:
            indices = [[2 for _ in range(max_length)] for _ in range(len(texts))]
            for i, text in enumerate(tqdm(texts)):
                for j, word in enumerate(text):
                    if word in self.word2idx:
                        indices[i][j] = self.word2idx[word]
                    else:
                        indices[i][j] = self.word2idx["<unk>"]   
                indices[i].insert(len(text), self.word2idx["<eos>"]) # Set <eos> at the end of the sentence
            if verbose:
                print("max_length:", max_length+1)
                
        
        return indices
    def save_vocab(self, file_path):
        """ 
        Saves the vocabulary to a pt file.
                params:
                    - str file_path: The path to the file where the vocabulary will be saved.
                returns:
                    None - The method saves the vocabulary to the specified file.
        """
        with open(file_path, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "word2count": self.word2count,
                "vocab_size": self.vocab_size
            }, f)

    
class Splitter:
    """ 
    This class splits a list of indices into training, validation and test sets.
        attributes:
            - float train_ratio: The ratio of the training set size to the total size of the dataset.
            - float val_ratio: The ratio of the validation set size to the total size of the dataset.
    """
    def __init__(self, train_ratio=0.70, val_ratio=0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, src_indices, tgt_indices):
        """ 
        Splits the indices into training, validation and test sets.
                params:
                    - List[list[int]] src_indices: The indices to split.
                    - List[list[int]] tgt_indices: The target indices to split.
                returns:
                    Tuple[Tuple[List[List[int]]]] - A tuple containing two tuples, each with three lists of indices:
        """
        src_train, src_test, tgt_train, tgt_test = train_test_split(
            src_indices, tgt_indices, test_size=1-self.train_ratio-self.val_ratio, random_state=42
        )
        src_train, src_val, tgt_train, tgt_val = train_test_split(
            src_train, tgt_train, test_size=self.val_ratio/(self.train_ratio+self.val_ratio), random_state=42
        )
        return (
            (src_train, src_val, src_test),
            (tgt_train, tgt_val, tgt_test)
        )

    def save_indices(self, indices, file_path):
        """ 
        Saves the indices to a pt file.
                params:
                    - List[List[int]] indices: The indices to save.
                    - str file_path: The path to the file where the indices will be saved.
                returns:
                    None - The method saves the indices to the specified file.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        torch.save(indices, file_path)