## **PyTorch Tensor Commands Summary **

### **1. Tensor Creation**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.randn(...)` | Create tensor with random normal values | `torch.randn(batch_size, hidden_dim, device=device)` |
| `torch.randint(...)` | Create tensor with random integers | `torch.randint(0, 10, (batch_size,), device=device)` |
| `torch.zeros(...)` | Create tensor filled with zeros | `torch.zeros(self.m, self.n, device=self.device, dtype=torch.float32)` |
| `torch.empty(...)` | Create uninitialized tensor | `torch.empty(self.N, dtype=torch.float32, device=self.device)` |
| `torch.empty_like(...)` | Create empty tensor with same shape | `torch.empty_like(self.host_tensor, device=self.device)` |
| `torch.stack(...)` | Stack sequence of tensors | `torch.stack(blocks)` |
| `torch.cat(...)` | Concatenate tensors along dimension | `torch.cat(self.microbatches[start:end], dim=0)` |

### **2. Tensor Operations**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.matmul(A, B)` | Matrix multiplication | `torch.matmul(self.A, self.B)` |
| `torch.sum(tensor)` | Sum all elements | `torch.sum(self.device_buffer)` |
| `tensor.mul_(scalar)` | In-place multiply by scalar | `self.gpu_data.mul_(2.0)` |
| `tensor.add_(scalar)` | In-place add scalar | `self.gpu_data.add_(1.0)` |
| `tensor.view(...)` | Reshape tensor | `self.device_data.view(torch.int32)` |
| `tensor.unsqueeze(dim)` | Add dimension | `torch.sum(self.device_buffer).unsqueeze(0)` |

### **3. Data Type Conversion**

| Command | Description | Example |
|---------|-------------|---------|
| `tensor.half()` | Convert to FP16 | `self.model = self.model.half()` |
| `tensor.float()` | Convert to FP32 | `self._verify_input = self.microbatches[0].float().clone()` |
| `tensor.to(dtype=...)` | Convert to specified dtype | `verify_input.to(dtype=model_params[0].dtype, device=self.device)` |

### **4. Memory & Device Management**

| Command | Description | Example |
|---------|-------------|---------|
| `tensor.to(device)` | Move tensor to device | `self.model.to(self.device)` |
| `torch.device("cuda")` | Create CUDA device | `self.device = torch.device("cuda")` |
| `tensor.pin_memory()` | Pin tensor memory for async transfer | `torch.randn(N, dtype=torch.float32).pin_memory()` |
| `torch.randn(..., pin_memory=True)` | Create pinned tensor directly | `torch.randn(num_elements, dtype=torch.float32, pin_memory=True)` |
| `tensor.contiguous()` | Make tensor contiguous in memory | `torch.randn(...).contiguous()` |
| `tensor.cpu()` | Move tensor to CPU | `self.cpu_data = self.gpu_data.cpu()` |
| `torch.cuda.empty_cache()` | Clear CUDA memory cache | `torch.cuda.empty_cache()` |

### **5. Data Transfer**

| Command | Description | Example |
|---------|-------------|---------|
| `tensor.copy_(src, non_blocking=False)` | Blocking copy | `self.device_data.copy_(self.host_data, non_blocking=False)` |
| `tensor.copy_(src, non_blocking=True)` | Async (non-blocking) copy | `self.device_data.copy_(self.host_data, non_blocking=True)` |

### **6. Tensor Cloning & Detaching**

| Command | Description | Example |
|---------|-------------|---------|
| `tensor.clone()` | Create copy of tensor | `self._verify_input = self.microbatches[0].clone()` |
| `tensor.detach()` | Detach from computation graph | `self.output.detach().clone()` |
| `tensor.detach().clone()` | Detach and clone (common pattern) | `self._impl.gpu_data[:1000].detach().clone()` |

### **7. CUDA Synchronization & Streams**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.cuda.synchronize()` | Synchronize all CUDA operations | `torch.cuda.synchronize()` |
| `torch.cuda.synchronize(device)` | Synchronize specific device | `torch.cuda.synchronize(self.device)` |
| `torch.cuda.Stream()` | Create CUDA stream | `self.copy_stream = torch.cuda.Stream()` |
| `torch.cuda.stream(stream)` | Context manager for stream | `with torch.cuda.stream(self.stream):` |
| `torch.cuda.current_stream()` | Get current stream | `self.compute_stream = torch.cuda.current_stream()` |
| `stream.synchronize()` | Synchronize specific stream | `self.stream.synchronize()` |
| `stream.wait_stream(other)` | Wait for another stream | `torch.cuda.current_stream().wait_stream(self.copy_stream)` |

### **8. Random Seeding**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.manual_seed(seed)` | Set CPU random seed | `torch.manual_seed(42)` |
| `torch.cuda.manual_seed_all(seed)` | Set CUDA random seed (all GPUs) | `torch.cuda.manual_seed_all(42)` |

### **9. CUDA Info & Properties**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.cuda.is_available()` | Check CUDA availability | `if torch.cuda.is_available():` |
| `torch.cuda.current_device()` | Get current GPU ID | `gpu_id = torch.cuda.current_device()` |
| `torch.cuda.get_device_properties(id)` | Get GPU properties | `props = torch.cuda.get_device_properties(0)` |

### **10. Backend Settings**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.backends.cuda.matmul.allow_tf32` | Check/set TF32 for matmul | `bool(torch.backends.cuda.matmul.allow_tf32)` |

### **11. Torch Compile**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.compile(fn, mode=...)` | Compile function for optimization | `torch.compile(matmul_fn, mode="reduce-overhead")` |

### **12. Neural Network Layers (torch.nn)**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.nn.Sequential(...)` | Create sequential container | `torch.nn.Sequential(nn.Linear(...), nn.ReLU(), ...)` |
| `torch.nn.Linear(in, out)` | Linear/Dense layer | `torch.nn.Linear(self.hidden_dim, self.hidden_dim)` |
| `torch.nn.ReLU()` | ReLU activation | `torch.nn.ReLU()` |
| `torch.nn.GELU()` | GELU activation | `nn.GELU()` |
| `model.parameters()` | Get model parameters | `sum(p.numel() for p in self.model.parameters())` |
| `model.eval()` | Set model to eval mode | `self.model.eval()` |

### **13. Loss & Functions**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.nn.functional.cross_entropy(...)` | Cross entropy loss | `torch.nn.functional.cross_entropy(logits, target)` |
| `nn.functional.mse_loss(...)` | Mean squared error loss | `nn.functional.mse_loss(out, target)` |
| `loss.backward()` | Backpropagation | `loss.backward()` |

### **14. Optimizers**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.optim.SGD(params, lr=...)` | SGD optimizer | `torch.optim.SGD(self.model.parameters(), lr=1e-3)` |
| `optim.Adam(params, lr=...)` | Adam optimizer | `optim.Adam(model.parameters(), lr=1e-3)` |
| `optimizer.zero_grad(set_to_none=True)` | Clear gradients (memory efficient) | `self.optimizer.zero_grad(set_to_none=True)` |
| `optimizer.step()` | Update parameters | `self.optimizer.step()` |

### **15. Context Managers**

| Command | Description | Example |
|---------|-------------|---------|
| `torch.no_grad()` | Disable gradient tracking | `with torch.no_grad():` |

### **16. Tensor Attributes**

| Command | Description | Example |
|---------|-------------|---------|
| `tensor.shape` | Get tensor shape | `self._verify_input.shape[0]` |
| `tensor.dtype` | Get tensor data type | `model_params[0].dtype == torch.float16` |
| `tensor.numel()` | Number of elements | `int(data_bits.numel())` |
| `tensor.device` | Get tensor device | `tensor.device` |

---

### **Key Patterns by Use Case**

**Memory-Optimized Transfers:**
```python
# Baseline (slow): pageable memory + blocking copy
host_data = torch.randn(N, dtype=torch.float32, pin_memory=False)
device_data.copy_(host_data, non_blocking=False)

# Optimized (fast): pinned memory + async copy
host_data = torch.randn(N, dtype=torch.float32, pin_memory=True)
device_data.copy_(host_data, non_blocking=True)
```

**Double-Buffering for Overlapping:**
```python
copy_stream = torch.cuda.Stream()
with torch.cuda.stream(copy_stream):
    device_buffers[slot].copy_(host_tensor, non_blocking=True)
torch.cuda.current_stream().wait_stream(copy_stream)
```

**Precision Optimization:**
```python
# FP16 for tensor cores
model = model.half()
data = torch.randn(..., dtype=torch.float16)
```

**Batch Fusion:**
```python
fused_batch = torch.cat(microbatches[start:end], dim=0)
fused_target = torch.cat(targets[start:end], dim=0)
```

