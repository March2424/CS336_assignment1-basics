from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]     # 获取与 p 关联的状态。
                t = state.get("t", 0)     # 从状态获取迭代次数，或初始值。
                grad = p.grad.data        # 获取损失关于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重张量。
                state["t"] = t + 1        # 增加迭代次数。

        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self,
                params,
                lr: float = 1e-3,
                betas: tuple = (0.9,0.99),
                eps: float = 1e-8,
                weight_decay: float = 0.01
                ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # state中存储每个参数的状态，步数，一阶矩m_t，二阶矩v_t.
                # Key 是真正的nn.Parameter对象本身，Value是另一个字典，专门用来存储这个参数的私有状态
                if len(state) == 0:
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg,exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad,alpha = 1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value = 1-beta2)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                v_t = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                m_t = exp_avg / bias_correction1

                step_size = lr * m_t / v_t
                # tensor.addcdiv_(tensor1, tensor2, *, value)
                # tensor = tensor + value*(tensor1/tensor2)
                p.data.mul_(1-lr * weight_decay)
                p.data.add_(step_size,alpha = -1)

        return loss


