import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor,torch.Tensor]:
    # 输出[batch_size,seq_len]分别为预测和答案

    start_indices = np.random.randint(0,len(dataset)-context_length,size = (batch_size,))

    offsets = np.arange(context_length+1)
    # 广播生成长度为[batch_size,context_length+1]
    block_indices = start_indices[:,None] + offsets

    data_blocks = torch.from_numpy(dataset[block_indices].astype(np.int64))

    x = data_blocks[:,:-1]
    y = data_blocks[:,1:]

    x,y = x.to(device),y.to(device)

    return x,y