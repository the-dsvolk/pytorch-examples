import asyncio
import torch
import torch.nn as nn

# ---- 1. Model ----
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a fully connected layer: 4 input features -> 2 output features
        # Linear transformation: y = xW^T + b where:
        # - x is the input with 4 features
        # - W is a learnable weight matrix of shape (2, 4) 
        # - b is a learnable bias vector of shape (2,)
        # - y is the output with 2 features
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        # Forward pass: defines how data flows through the model
        # This method is automatically called when you do model(input)
        # Note: self.fc(x) works because nn.Linear implements __call__ method
        # So self.fc(x) actually calls self.fc.__call__(x) which runs the linear layer
        return self.fc(x)

# Create model instance and move it to GPU for faster computation
model = SimpleModel()
# Use DataParallel for multi-GPU support if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to("cuda")
# Set model to evaluation mode (disables dropout, batch norm training behavior)
# Important for inference to get consistent, deterministic results
model.eval()

# ---- 2. Inference ----
def run_inference(batch_x):
    # Disable gradient computation for inference (saves memory and speeds up computation)
    # We don't need gradients since we're not training, just doing forward pass
    with torch.no_grad():
        # Move batch data from CPU to GPU memory for faster computation
        batch_x = batch_x.to("cuda")
        batch_y = model(batch_x)
        # CRITICAL: Move results from CUDA GPU memory back to CPU memory
        # This is essential because:
        # 1. batch_y currently exists only in GPU memory and cannot be used by CPU operations
        # 2. Without .cpu(), you'd get "RuntimeError: Expected all tensors to be on the same device"
        # 3. CPU memory is needed for printing, saving, or any non-CUDA operations
        # 4. Creates a copy in CPU memory while preserving the original GPU computation benefits
        return batch_y.cpu()

# ---- 3. Producer ----
async def producer(queue, n_requests=10):
    for i in range(n_requests):
        x = torch.randn(1, 4)  # one request
        await queue.put((i, x))
        print(f"[Producer] queued request {i}")
        await asyncio.sleep(0.05)  # simulate delay

# ---- 4. Batching Consumer ----
async def batching_consumer(queue, batch_size=4, timeout=0.2):
    shutdown_signal = False
    
    while not shutdown_signal:
        batch = []
        ids = []

        try:
            # Wait for at least one request
            first = await asyncio.wait_for(queue.get(), timeout=timeout)
            if first is None:  # shutdown signal
                shutdown_signal = True
                break
            ids.append(first[0])
            batch.append(first[1])

            # Collect more requests up to batch_size
            while len(batch) < batch_size:
                try:
                    req_id, x = queue.get_nowait()
                    if req_id is None:
                        shutdown_signal = True
                        break
                    ids.append(req_id)
                    batch.append(x)
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            # Timeout with empty queue → continue trying
            continue

        # Skip processing if no requests collected
        if not batch:
            continue

        # Stack into one batch tensor using torch.cat
        # Shape notation: (rows, columns) = (num_samples, features_per_sample)
        # Example: 4 tensors of shape (1,4) -> one tensor of shape (4,4)
        # Individual: [[1.2, -0.5, 0.8, 2.1]] (1 row, 4 cols) = (1,4)
        # Batched:    [[1.2, -0.5, 0.8, 2.1],  (4 rows, 4 cols) = (4,4)
        #              [0.3,  1.7, -1.2, 0.9],
        #              [-0.8, 2.3,  0.4, -1.5],
        #              [1.9, -0.2,  3.1,  0.6]]
        # torch.cat concatenates along dim=0 (batch dimension)
        batch_x = torch.cat(batch, dim=0)
        results = await asyncio.to_thread(run_inference, batch_x)

        # Split results back per request
        # Important: Model preserves order! Input batch[0] -> results[0], batch[1] -> results[1], etc.
        # We only send tensors to model, not IDs, and rely on positional matching
        for req_id, res in zip(ids, results):
            print(f"[BatchConsumer] result for request {req_id}:", res)

# ---- 5. Orchestrator ----
async def main():
    # Use asyncio.Queue (not deque) for thread-safe async communication
    # asyncio.Queue provides: await queue.get(), await queue.put(), proper blocking/signaling
    # deque would require manual synchronization and wouldn't work with async/await
    queue = asyncio.Queue()

    # create_task() schedules the coroutine to run concurrently (doesn't block)
    # Without create_task(), producer() would run first and block until complete
    producer_task = asyncio.create_task(producer(queue, n_requests=10))
    consumer_task = asyncio.create_task(batching_consumer(queue, batch_size=4))

    await producer_task
    await queue.put(None)  # poison pill
    await consumer_task

if __name__ == "__main__":
    asyncio.run(main())