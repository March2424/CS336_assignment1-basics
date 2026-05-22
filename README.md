# CS336 Spring 2025 Assignment 1: Basics

## 	My implement of model

This is a baseline with 64 batch_size and 6e-4 lr.

```
python3 cs336_basics/train.py \
    --train_data_path data/tinystories/train.bin \
    --valid_data_path data/tinystories/val.bin \
    --run_name "baseline_bs256_lr6e-4" \
    --vocab_size 10000 \
    --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
    --max_iters 7000 \
    --batch_size 64 \
    --context_length 256 \
    --lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 700 \
    --out_dir model_result/TinyStories_baseline \
    --wandb_project "cs336-pertrained-tinystories" \
    --device cuda
```

and with the same para, I add the QKNorm、weight tying and Initialize projections of attention and ffn's output layers to zero. 

In another model, I add the MoE module to previous QKNorm and zero module.

### Learning curves

Here is an error!!! The batchsize in this pictures is bas256, but in this experient it's 64!

![W&B Chart 2026_5_22 17_00_02](.\image\W&B Chart 2026_5_22 17_00_02.png)

![W&B Chart 2026_5_22 17_00_19](.\image\W&B Chart 2026_5_22 17_00_19.png)



Finally Obtained losses of 1.6458, 1.4713, and 1.4596 respectively

### Why QKNorm and zero Init model's performance of the model is not significantly different from the baseline?

1. because the low learning rate, QKNorm and zero init can't make a difference to the performance, 
2. weight tying reduce the para capacity, in the limited iters,  model's ability to fit complex contexts in TinyStories is subject to physical limitations.



### Why  MoE got the poor performance in tinystories datasets?

1. MoE Router need big batch_size to get the load balancing, but this experience only get 64 batch size, so this can affect the aux loss and the affect the performance of model. 
2. Since only the top-K (such as top-2) experts are activated each time, the model lacks a globally shared module to learn the frequent punctuation marks and common grammar that exist in all the data.
3. Perhaps we can adopt the Shared Expert architecture of Deepseek. However, this also requires consideration of the balance between memory and performance. Its necessity is not significant for small models.

Then I tried to increase the learning rate to 3e-4, and set the batch size to 128.

```
python3 cs336_basics/train.py \
    --train_data_path data/tinystories/train.bin \
    --valid_data_path data/tinystories/val.bin \
    --run_name "updated_bs256_baseline" \
    --vocab_size 10000 \
    --num_layers 4 --num_heads 16 --d_model 512 --d_ff 1344 \
    --max_iters 7000 \
    --batch_size 128 \
    --context_length 256 \
    --lr 3e-3 \
    --min_lr 3e-4 \
    --warmup_iters 700 \
    --out_dir model_result/TinyStories_BS256_baseline \
    --wandb_project "cs336_tinystories" \
    --device cuda
```



![W&B Chart 2026_5_22 17_52_36](.\image\W&B Chart 2026_5_22 17_52_36.png)

![W&B Chart 2026_5_22 17_52_54](.\image\W&B Chart 2026_5_22 17_52_54.png)

**Updated Dense (nomoe)**: **1.297** 

**Standard Baseline (baseline)**: **1.353**

**MoE Variant (moe)**: **1.544**

SO, with a very small number of parameters (such as only 4 layers and 512 dimensions) and a conservative learning rate, forcibly splitting the parameters to establish a hybrid expert model will lead to a significant performance degradation. It is not recommended to add the MoE module.











