import time
import tracemalloc
import os
import json
from tqdm import tqdm

from cs336_basics.bpe import train_bpe 

def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_prefix: str, save_dir: str):
    """
    将 BPE 训练结果保存为工业标准的 vocab.json 和 merges.txt
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # 1. 保存 vocab.json
    # 工业标准的 vocab.json 通常是 { "token字符串": ID } 的映射
    # ==========================================
    vocab_dict = {}
    for token_id, token_bytes in vocab.items():
        # 将 bytes 解码为 string，遇到非法 utf-8 字节用标准的替换符替代，防止 JSON 报错
        token_str = token_bytes.decode("utf-8", errors="replace")
        vocab_dict[token_str] = token_id
        
    vocab_path = os.path.join(save_dir, f"{output_prefix}_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 可以让 JSON 文件里直接显示中文/特殊符号，而不是 \uXXXX
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
    # ==========================================
    # 2. 保存 merges.txt
    # 工业标准的 merges.txt 每行是一个合并规则，用空格隔开
    # ==========================================
    merges_path = os.path.join(save_dir, f"{output_prefix}_merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        # 很多标准实现会在 merges.txt 第一行加上版本说明
        f.write("# version: 1.0\n") 
        for pair in merges:
            p1 = pair[0].decode("utf-8", errors="replace")
            p2 = pair[1].decode("utf-8", errors="replace")
            # 用空格拼接写入
            f.write(f"{p1} {p2}\n")
            
    print(f"[结果保存] 词表和合并规则已成功保存至:\n  - {vocab_path}\n  - {merges_path}")

def run_bpe_experiment(input_path: str, vocab_size: int, special_tokens: list[str], output_prefix: str):
    print(f"========== 开始在 {input_path} 上训练 BPE ==========")
    
    # 开始记录内存和时间
    tracemalloc.start()
    start_time = time.time()
    
    # 运行 BPE 训练
    vocab, merges = train_bpe(
        input_path=input_path, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens,
        num_processes=4 # 根据你的 Mac 核心数调整
    )
    
    # 停止记录
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_minutes = (end_time - start_time) / 60
    peak_memory_gb = peak / (1024 ** 3)
    
    print(f"[资源消耗]")
    print(f"训练耗时: {time_minutes:.2f} 分钟")
    print(f"内存峰值: {peak_memory_gb:.2f} GB")
    
    # 寻找最长的 Token
    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')
    
    print(f"\n[词表分析]")
    print(f"最长的 Token 长度: {len(longest_token_bytes)} 字节")
    print(f"最长的 Token 内容: {repr(longest_token_str)}")
    
    # ----------------------------------------------------
    # 替换了原有的 pickle 保存逻辑，使用新的 json/txt 保存函数
    # ----------------------------------------------------
    save_dir = "data/experiment_results"
    save_vocab_and_merges(vocab, merges, output_prefix, save_dir)
    print("====================================================\n")

if __name__ == "__main__":

    run_bpe_experiment(
        input_path="data/TinyStoriesV2-GPT4-train.txt", 
        vocab_size=10_000, 
        special_tokens=["<|endoftext|>"],
        output_prefix="tinystories"
    )
    
    run_bpe_experiment(
        input_path="data/owt_train.txt", 
        vocab_size=32_000, 
        special_tokens=["<|endoftext|>"],
        output_prefix="owt"
    )