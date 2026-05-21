import os
import numpy as np
from tokenizers import Tokenizer
import argparse

def process_file_efficiently(input_txt_path, output_bin_path, tokenizer, batch_size=4096):
    """
    高效、低内存占用的数据转换函数
    """
    print(f"开始处理: {input_txt_path}")
    if not os.path.exists(input_txt_path):
        print(f"错误: 找不到文件 {input_txt_path}")
        return

    # 1. 先用追加模式打开或创建二进制文件，避免在内存中堆积巨大的列表
    # 'wb' 会清空原文件重写
    with open(output_bin_path, 'wb') as bin_file:
        batch_texts = []
        total_tokens = 0
        line_count = 0

        # 逐行流式读取，内存占用极低
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    batch_texts.append(text)
                
                # 当累积到一定数量时，进行批量多线程 Tokenize
                if len(batch_texts) >= batch_size:
                    # encode_batch 会自动调用 C++ 底层的高并发多线程
                    encodings = tokenizer.encode_batch(batch_texts)
                    
                    # 提取所有 ID 并展平
                    ids = []
                    for enc in encodings:
                        ids.extend(enc.ids)
                    
                    # 转换为 uint16 并直接写入硬盘文件
                    if ids:
                        np_ids = np.array(ids, dtype=np.uint16)
                        bin_file.write(np_ids.tobytes())
                        total_tokens += len(ids)
                    
                    line_count += len(batch_texts)
                    print(f"已处理文本行: {line_count}, 已写入 Token 数: {total_tokens}", end='\r')
                    batch_texts = [] # 清空批次，释放内存

            # 处理最后一批尾部剩余的数据
            if batch_texts:
                encodings = tokenizer.encode_batch(batch_texts)
                ids = []
                for enc in encodings:
                    ids.extend(enc.ids)
                if ids:
                    np_ids = np.array(ids, dtype=np.uint16)
                    bin_file.write(np_ids.tobytes())
                    total_tokens += len(ids)
                line_count += len(batch_texts)

    print(f"\n处理完成！共处理行数: {line_count}, 总计 Token 数量: {total_tokens}")
    print(f"二进制文件已保存至: {output_bin_path}\n" + "-"*30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="./data/openwebtext/tokenizer.json")
    parser.add_argument("--train_input", type=str, default="./data/owt_train.txt")
    parser.add_argument("--valid_input", type=str, default="./data/owt_valid.txt")
    parser.add_argument("--train_output", type=str, default="./data/openwebtext/train.bin")
    parser.add_argument("--valid_output", type=str, default="./data/openwebtext/val.bin")
    # 这里的 batch_size 指的是多少行文本一起喂给 Tokenizer，可以根据 CPU 核心数调整
    parser.add_argument("--batch_size", type=int, default=8192, help="每次并行处理的文本行数")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--tokenizer_path", type=str, default="./data/tinystories/tokenizer.json")
    # parser.add_argument("--train_input", type=str, default="./data/TinyStoriesV2-GPT4-train.txt")
    # parser.add_argument("--valid_input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    # parser.add_argument("--train_output", type=str, default="./data/tinystories/train.bin")
    # parser.add_argument("--valid_output", type=str, default="./data/tinystories/val.bin")
    # # 这里的 batch_size 指的是多少行文本一起喂给 Tokenizer，可以根据 CPU 核心数调整
    # parser.add_argument("--batch_size", type=int, default=8192, help="每次并行处理的文本行数")
    args = parser.parse_args()

    print(f"加载 Tokenizer: {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"当前词表大小: {vocab_size}")
    if vocab_size > 65535:
        raise ValueError("词表大小超过 65535，无法使用 np.uint16 存储！")

    # 执行优化后的处理流程
    process_file_efficiently(args.train_input, args.train_output, tokenizer, args.batch_size)
    process_file_efficiently(args.valid_input, args.valid_output, tokenizer, args.batch_size)

if __name__ == "__main__":
    main()