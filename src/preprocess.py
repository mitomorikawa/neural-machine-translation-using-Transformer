import library.preprocessor as preprocessor

src_language = "eng"
tgt_language = "fra"

file_name = f"{src_language}_{tgt_language}.csv"

print("Loading data...")
dataloader = preprocessor.DataLoader(f"../data/{file_name}")
src_texts, tgt_texts = dataloader.load()
print("Done loding data. Number of sentences:", len(src_texts), "\n\n")

print("Standardizing texts...")
standardizer = preprocessor.Standardizer()
standadized_src_texts = standardizer.standardize(src_texts)
print("Done standardizing source language texts.")
standadized_tgt_texts = standardizer.standardize(tgt_texts)
print("Done standardizing target language texts.\n\n")

print("Tokenizing texts...")
tokenizer = preprocessor.Tokenizer()
src_tokens, tgt_tokens = tokenizer.word_tokenize(standadized_src_texts, standadized_tgt_texts, 64)
print("Done tokenizing source language texts. First 10 sentences:", src_tokens[:10], "\n\n")
print("Done tokenizing target language texts. First 10 sentences:", tgt_tokens[:10], "\n\n")

print("Indexing source language texts...")
src_indexer = preprocessor.Indexer()
src_indexer.build_vocab(src_tokens)
print("Done indexing source language texts. ")
print("Vocabulary size:", src_indexer.vocab_size)
print("First_10 words in source language vocabulary:", list(src_indexer.word2idx.keys())[:10], "\n\n")

print("Indexing target language texts...")
tgt_indexer = preprocessor.Indexer()
tgt_indexer.build_vocab(tgt_tokens)
print("Done indexing target language texts. ")
print("Vocabulary size:", tgt_indexer.vocab_size)
print("First 10 words in target language vocabulary:", list(tgt_indexer.word2idx.keys())[:10], "\n\n")
print("Saving vocabulary...")
src_indexer.save_vocab(f"../data/{src_language}_vocab.pkl")
tgt_indexer.save_vocab(f"../data/{tgt_language}_vocab.pkl")
print("Done saving vocabulary to ../data/{src_language}_vocab.pkl and ../data/{tgt_language}_vocab.pkl.\n\n")


print("Converting source language tokens to indices...")
src_indices = src_indexer.text_to_indices(src_tokens)
print("Done converting source language tokens to indices. First 10 sentences:", src_indices[:10], "\n\n")
print("Converting target language tokens to indices...")
tgt_indices = tgt_indexer.text_to_indices(tgt_tokens, prepend_sos=True)
print("Done converting target language tokens to indices. First 10 sentences:", tgt_indices[:10], "\n\n")

splitter = preprocessor.Splitter()
print("Splitting indices...")
((src_train, src_val, src_test),(tgt_train, tgt_val, tgt_test)) = splitter.split(src_indices, tgt_indices)
print("Done splitting indices.")
print(f"Training set size: {len(src_train)} sentences.")
print(f"Validation set size: {len(src_val)} sentences.")
print(f"Test set size: {len(src_test)} sentences.\n\n")



print("Saving training data...")
splitter.save_indices(src_train, f"../data/{src_language}_train.pt")
splitter.save_indices(tgt_train, f"../data/{tgt_language}_train.pt")
print(f"Done saving training data to ../data/{src_language}_train.pt and ../data/{tgt_language}_train.pt.\n\n")

print("Saving validation data...")
splitter.save_indices(src_val, f"../data/{src_language}_val.pt")
splitter.save_indices(tgt_val, f"../data/{tgt_language}_val.pt")
print(f"Done saving validation data to ../data/{src_language}_val.pt and ../data/{tgt_language}_val.pt.\n\n")

print("Saving test data...")
splitter.save_indices(src_test, f"../data/{src_language}_test.pt")
splitter.save_indices(tgt_test, f"../data/{tgt_language}_test.pt")
print(f"Done saving test data to ../data/{src_language}_test.pt and ../data/{tgt_language}_test.pt.\n\n")