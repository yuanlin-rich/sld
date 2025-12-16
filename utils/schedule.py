# 扩散模型采样过程的动态调度优化

import torch
import warnings

def get_fast_schedule(origial_timesteps, fast_after_steps, fast_rate):
    # 快速调度策略
    if fast_after_steps >= len(origial_timesteps) - 1:
        return origial_timesteps
    new_timesteps = torch.cat((origial_timesteps[:fast_after_steps], origial_timesteps[fast_after_steps+1::fast_rate]), dim=0)
    return new_timesteps

def dynamically_adjust_inference_steps(scheduler, index, t):
    # 动态调整推理步数
    prev_t = scheduler.timesteps[index+1] if index+1 < len(scheduler.timesteps) else -1
    scheduler.num_inference_steps = scheduler.config.num_train_timesteps // (t - prev_t)
    if index+1 < len(scheduler.timesteps):
        if scheduler.config.num_train_timesteps // scheduler.num_inference_steps != t - prev_t:
            warnings.warn(f"({scheduler.config.num_train_timesteps} // {scheduler.num_inference_steps}) != ({t} - {prev_t}), so the step sizes may not be accurate")
    else:
        # as long as we hit final cumprob, it should be fine.
        if scheduler.config.num_train_timesteps // scheduler.num_inference_steps > t - prev_t:
            warnings.warn(f"({scheduler.config.num_train_timesteps} // {scheduler.num_inference_steps}) > ({t} - {prev_t}), so the step sizes may not be accurate")
