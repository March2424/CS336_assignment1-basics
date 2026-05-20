import argparse
import os
import torch
import numpy as np
import wandb
import typing

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, gradient_clip
from cs336_basics.optimizer import cosine_annealing_lr
from cs336_basics.dataloader import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.nn_utils import cross_entropy

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)

    # --- 实验/消融 (Ablation) 开关 ---
    # Ablation 1: 移除 RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm completely")
    # Ablation 2: Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre", "post"], help="Normalization placement")
    # Ablation 3: 移除 RoPE (NoPE)
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings")
    # Ablation 4: SwiGLU vs SiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Type of Feed-Forward Network")

   
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
   
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')

    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    
    
    actual_rope_theta = None if args.no_rope else 10000.0
    
    use_rms_norm = not args.no_rms_norm

    # # 3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        device=args.device,
        # 传入实验参数
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}")

   
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # # 6. 初始化 WandB 监控
    wandb.init(
        project="cs336-assignment1",
        name=args.run_name,
        config=args
    )

    # 分为两种循环方法，一种用epoch，一种用maxiters
    for it in range(start_iter, args.max_iters):
        
        lr = cosine_annealing_lr(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        
        gradient_clip(model.parameters(), args.max_norm)
        optimizer.step()

        
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": v_loss.item(),
                    "lr": lr,
                    "iter": it + 1
                })

        # # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    
    final_ckpt_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_ckpt_path)
    print(f"训练结束，最终模型已保存至: {final_ckpt_path}")
    wandb.finish()

if __name__ == "__main__":
    main()