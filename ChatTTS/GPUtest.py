import torch
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print(torch.__version__)  # Should print the version of PyTorch
print(torch.version.cuda)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.current_device())  # Should print: 0 (or another GPU ID)
print(torch.cuda.device_count())  # Should print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU
if torch.cuda.is_available():
    print("Current CUDA Device: ", torch.cuda.current_device())  # Should print: 0 (or another GPU ID)
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
a = True
if(a == True):
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"PyTorch version: {torch.__version__}")
        
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        for i in range(num_gpus):
            # Get the name of the GPU
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            # Get the CUDA compute capability
            compute_capability = torch.cuda.get_device_capability(i)
            print(f"Compute capability of GPU {i}: {compute_capability}")
            
            # Get the current memory allocated and cached
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"Memory allocated on GPU {i}: {allocated:.2f} GB")
            print(f"Memory cached on GPU {i}: {cached:.2f} GB")
    else:
        print("CUDA is not available.")
    a = False