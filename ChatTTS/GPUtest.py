import torch
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.current_device())  # Should print: 0 (or another GPU ID)
print(torch.cuda.device_count())  # Should print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU
