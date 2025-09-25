import torch
torch.cuda.empty_cache()  # 清空未使用的 GPU 缓存
torch.cuda.reset_peak_memory_stats()  # 重置显存统计信息（可选）