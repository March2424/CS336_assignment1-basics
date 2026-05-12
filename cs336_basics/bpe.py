import os
from collections import defaultdict, Counter
import regex as re
import json
from multiprocessing import Pool, get_context
from typing import BinaryIO
from tqdm import tqdm

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)

def init_vocab(special_tokens:list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int,bytes] = {x:bytes([x]) for x in range(256)}
    cur_length = 256
    if special_tokens:
        for token in special_tokens:
            byte_token = token.encode("utf-8")
            vocab[cur_length] = byte_token
            cur_length+=1
    return vocab

# # Pre_processing
# def split_by_specialtoken(text: str, special_tokens: list[str], 
#                           include_special: bool = False) -> list[str]:
#     if not special_tokens:
#         return [text]
    
#     special_tokens_sorted = sorted(special_tokens,key = len, reverse=True)
#     pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

#     if include_special:
#         res_chunks = re.split(f"({pattern})", text)
#     else:
#         res_chunks = re.split(pattern,text)
#     return res_chunks

# def Pre_Tokenization(unsplit_text: str,special_tokens: list[str], 
#                      including_special: bool = False) -> Counter:
#     raw_counts = Counter()
#     chunks = split_by_specialtoken(unsplit_text,special_tokens,include_special=including_special)
#     for chunk in chunks:
#         if including_special and chunk in special_tokens:
#             continue
#         else:
#             for m in re.finditer(_COMPILED_PAT, chunk):
#                 word = m.group(0)
#                 raw_counts[tuple(bytes([t]) for t in word.encode("utf-8"))] += 1
#     return raw_counts
                
            


def train_bpe(input_path: str, # 输入文件的路径
              vocab_size: int,  # 词表大小 256 + n
              special_tokens: list[str], # 特殊token列表
              num_processes: int = 4,
              num_chunks: int = 200
) -> tuple[dict[int,bytes], list[tuple[bytes,bytes]]]: 
# vocab分词器词表，一个从 int（词表中的 token ID）到 bytes（token 字节）的映射。
# merges训练产生的 BPE 合并列表。每个列表项是一个包含2个 bytes 的元组 
# (<token1>, <token2>)，表示 <token1> 被合并到了 <token2>。这些合并操作应按创建的顺序排列。
    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)
    endoftext_bytes = "<|endoftext|>".encode("utf-8")
    with open(input_path,"rb") as f:
        bounds = find_chunk_boundaries(f, num_chunks, endoftext_bytes)

    task_args = [
        (input_path,start,end,special_tokens)
        for start, end in zip(bounds[:-1],bounds[1:])
    ]
    with get_context("forkserver").Pool(processes=num_processes) as pool:
        chunk_counters = pool.map(process_chunk, task_args)

    global_counts = Counter()

    with get_context("forkserver").Pool(processes=num_processes) as pool:
        # imap_unordered 会在某个 chunk 处理完后立刻 yield 结果
        for chunk_counter in tqdm(pool.imap_unordered(process_chunk, task_args), total=len(task_args)):
            # 立即合并到主字典
            global_counts.update(chunk_counter)
            # 立即销毁子字典，释放宝贵的内存！
            del chunk_counter 
            

    
    word_list = []
    count_list = []
    for w,f in global_counts.items():
        word_list.append(list(w))
        count_list.append(f)
    # pair_state存储所有相邻字节对和出现频率，ids存储需要合并的字节对出现在word_list的下标集合
    pair_state = defaultdict(int)
    ids = defaultdict(set)
    # 初始化
    for index,word in enumerate(word_list):
        freq = count_list[index]
        for i in range(len(word)-1):
            pair = (word[i],word[i+1])
            pair_state[pair] += freq
            ids[pair].add(index) 
    merges = []

    for _ in tqdm(range(num_merges), desc="BPE Merging", total=num_merges, smoothing=0.1):
        if not pair_state:
            break
        best_pair = max(pair_state.items(), key = lambda x: (x[1],x[0]))[0]

        merges.append(best_pair)
        new_token = best_pair[0]+best_pair[1]

        affected_index = ids[best_pair].copy()

        for j in affected_index:
            word = word_list[j]
            freq = count_list[j]
            i = 0
            while i < len(word)-1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    if i > 0:
                        prev_pair = (word[i-1],word[i])
                        pair_state[prev_pair] -= freq
                        if pair_state[prev_pair] == 0:
                            del pair_state[prev_pair]
                            del ids[prev_pair
                                    ]
                    if i < len(word) - 2:
                        next_pair = (word[i+1],word[i+2])
                        pair_state[next_pair]-= freq
                        if pair_state[next_pair] == 0:
                            del pair_state[next_pair]
                            del ids[next_pair]

                    word[i] = new_token
                    del word[i+1]

                    if i > 0:
                        new_prev = (word[i-1],word[i])
                        pair_state[new_prev] += freq
                        ids[new_prev].add(j)

                    if i < len(word) - 1:
                        new_next = (word[i],word[i+1])
                        pair_state[new_next] += freq
                        ids[new_next].add(j)
                else:
                    i+=1
        del pair_state[best_pair]
        del ids[best_pair]
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
    
    return vocab, merges
        
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        documents = re.split(pattern, chunk)
    else:
        documents = [chunk]
    
    raw_counts = Counter()
    for doc in documents:
        if not doc:
            continue
        else:
            for m in re.finditer(_COMPILED_PAT, doc):
                word = m.group(0)
                raw_counts[tuple(bytes([t]) for t in word.encode("utf-8"))] += 1

    return raw_counts



