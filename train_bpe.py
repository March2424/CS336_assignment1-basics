import time
import tracemalloc
import pickle
import os
from tqdm import tqdm


from cs336_basics.bpe import train_bpe 

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
    

    os.makedirs("data/experiment_results", exist_ok=True)
    

    with open(f"data/experiment_results/{output_prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        

    with open(f"data/experiment_results/{output_prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

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