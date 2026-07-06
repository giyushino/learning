"""
my bpe implementation 
"""

import time
import os
import regex as re
import multiprocessing as mp

from collections import defaultdict

from .pretokenization_example import find_chunk_boundaries

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


def get_pair_counts(word):
    counts = {}
    for i in range(len(word) - 1):
        p = (word[i], word[i + 1])
        counts[p] = counts.get(p, 0) + 1

    return counts


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


def run_train_bpe(
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

    word_counts = {}
    for result in results:
        for word, count in result.items():
            word_counts[word] = word_counts.get(word, 0) + count

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf8')

    merges = []
    pair_counts = {}
    pair_to_word = defaultdict(set)

    for chunk_seq, freq in word_counts.items():
        for i in range(len(chunk_seq) - 1):
            pair = (chunk_seq[i], chunk_seq[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_word[pair].add(chunk_seq)

    
    for _ in range(vocab_size - len(vocab)):
        max_pair, _ = max(
            pair_counts.items(),
            key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]])
        )
        new_id = len(vocab)
        vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        for word in pair_to_word[max_pair].copy():
            freq = word_counts[word]
            old_pair_counts = get_pair_counts(word)

            new_word = tuple(merge_pairs(word, max_pair, new_id))
            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            new_pair_counts = get_pair_counts(new_word)

            for pair, count in old_pair_counts.items():
                pair_counts[pair] = pair_counts.get(pair, 0) - count * freq
                pair_to_word[pair].discard(word)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair)
                    pair_to_word.pop(pair)
            
            for pair, count in new_pair_counts.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count * freq
                pair_to_word[pair].add(new_word)
               

    return vocab, merges


if __name__ == "__main__":
    input_path = "/home/allan/nvim/learning/src/learning/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    run_train_bpe(input_path, vocab_size, special_tokens)

 
