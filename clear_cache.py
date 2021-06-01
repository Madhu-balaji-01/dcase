# from GPUtil import showUtilization as gpu_usage
import torch
from numba import cuda
import gc

print(torch.cuda.is_available())

# gc.collect()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)
torch.cuda.empty_cache()
# gpu_usage()   
