"""
my bpe implementation 
"""

import time
import os
import regex as re
import multiprocessing as mp

from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def record_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} took {t1 - t0} seconds to run")
        return result
    return wrapper


def merge_pairs(ids, pair, new_id):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2 
        else:
            new_ids.append(ids[i])
            i += 1

    return tuple(new_ids)


def pretokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as file:
        file.seek(start)
        chunk_text = file.read(end - start).decode('utf-8')

    split_pattern = "|".join(re.escape(tok) for tok in special_tokens)

    segments = re.split(split_pattern, chunk_text)
    word_count = {}

    for segment in segments:
        pretokenized_chunks = re.finditer(PAT, segment)

        for match in pretokenized_chunks:
            word = tuple(match.group().encode('utf-8'))
            word_count[word] = word_count.get(word, 0) + 1

    return word_count


@record_time
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'rb') as file: 
        num_processes = 4
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    args = [(input_path, boundaries[i], boundaries[i+1], special_tokens) for i in range(len(boundaries) - 1)]
    with mp.Pool(4) as pool:
        results = pool.map(pretokenize_chunk, args)

    word_count = {}
    for result in results:
        for word, count in result.items():
            word_count[word] = word_count.get(word, 0) + count

        
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for _ in range(vocab_size - len(vocab) - len(special_tokens)):
        seq_pair_count = {}
        for chunk_seq, freq in word_count.items():
            for i in range(len(chunk_seq) - 1):
                pair = (chunk_seq[i], chunk_seq[i+1])
                seq_pair_count[pair] = seq_pair_count.get(pair, 0) + freq
        
        max_pair, _ = max(
            seq_pair_count.items(),
            key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]])
        )
        new_id = len(vocab)
        vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        new_word_count = {}
        for chunk_seq, freq in word_count.items():
            merged = merge_pairs(chunk_seq, max_pair, new_id)
            new_word_count[merged] = new_word_count.get(merged, 0) + freq
        word_count = new_word_count

    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf8')

    return vocab, merges


if __name__ == "__main__":
    input_path = "/home/allan/nvim/learning/src/learning/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 258
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)

 
