import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int,
                out_features: int,
                device: torch.device | None=None,
                dtype: torch.dtype | None = None):
        # 继承父类
        super().__init__()
        kwargs = {'device': device,'dtype':dtype}
        self.in_features = in_features
        self.out_features = out_features
        # 先分配内存，因为nn.parameter必须包装一个已经存在的tensor,empty只要求一块内存不写入,不使用bias
        self.weight = nn.Parameter(torch.empty((out_features,in_features),**kwargs))

        std = math.sqrt(2.0 / (self.in_features+self.out_features))
        nn.init.trunc_normal_(self.weight,mean = 0.0,std = std,a=-3*std,b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch，seq,in] @ weight[out,in]T
        return torch.einsum('...i, oi -> ...o',x,self.weight)

class embedding(nn.Module):
    # num_embeddings: int为vocab_size embedding_dim: int是嵌入向量的维度即d_model
    def __init__(self, num_embeddings: int, embedding_dim: int,
                device: torch.device | None=None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        kwargs = {'device': device,'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**kwargs))
        nn.init.trunc_normal_(self.weight,mean = 0.0, std = 1,a=-3.0,b=3.0)
        

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_id[batch_size,seq_len]
        # 从weight[vocab_size,d_model]中lookup， 输出[batch_size,seq,d_model]
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        kwargs = {'device': device,'dtype':dtype}
        self.weight = nn.Parameter(torch.ones(d_model,**kwargs))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[batch_size,seq_len,d_model] * [d_model,1]逐元素相乘
        '''
        关于并行处理的想法：让batchsize的句子里的所有seqlen大小的词乘以了self.weight
        算出一个总的loss，在前向传播中weight被广播到了[batch_size,seq_len,d_model]的形状
        所以最终需要把所有分支梯度求和，最终self.weight.grad仍然为[d_model] 
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        rms_result = x / result
        return (rms_result * self.weight).to(in_dtype)


        



    
