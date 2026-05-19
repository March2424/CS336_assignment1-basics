import torch
from typing import Iterable

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    # logits[batch_size,seq_len,vocab_size],targets[batch_size,seq_len]
    # 1. 找到最大值 M 并平移 logits (数值稳定性)
    # 2. 提取 targets 对应的正确位置的 logits
    # 3. 计算 LogSumExp 项
    # 4. 计算每一个 token 的 loss
    # 5. 返回所有 token loss 的平均值 (标量 Tensor)
    M = torch.max(logits,dim = -1, keepdim=True).values

    z = logits - M
    # [batch_size,seq_len]
    targets_logits = torch.gather(logits,dim = -1,index = targets.unsqueeze(-1)).squeeze(-1)

    lse = M.squeeze(-1) + torch.log(torch.sum(torch.exp(z),dim = -1))

    loss = lse -  targets_logits

    return torch.mean(loss)

def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)