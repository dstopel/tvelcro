# Learning PyTorch / CUDA / vLLM

> **For a fresh Claude session:** this is a self-paced learning track, not an
> automated job. Work through it with the user one track at a time, running the
> commands together on a play pod and explaining results. Do not touch the
> `tvelcro-data` network volume while the T-VELCRO experiment queue is running
> (see `docs/EXPERIMENTS.md`); deploy a separate pod without the volume.
> The user is an emulation engineer with a physics M.Sc. and knows transformers,
> attention, and profiling at the FLOP level already. Skip ML basics. The value
> here is systems: kernels, memory bandwidth, and why FLOP reductions do not
> translate linearly into wall-clock time. That question is the open one in the
> user's own DATE paper, so tie lessons back to it.

## 0. Setup

Deploy a play pod (CUDA 13 host and image, no network volume):

```bash
cd ~/tvelcro
export RUNPOD_API_KEY=<key>
bash scripts/deploy_pod.sh            # A5000, CUDA 13, 60 GB disk
# prints: ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip>
# stop it when done:  STOP=<podid> bash scripts/deploy_pod.sh
```

On the pod:

```bash
nvidia-smi                                  # driver >= 580 means CUDA 13
nvcc --version                              # toolkit should also be 13.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install vllm                            # matches the CUDA 13 driver
git clone https://github.com/dstopel/tvelcro && bash tvelcro/pod_env/bootstrap.sh
```

Always run long jobs inside `tmux` (model loads take minutes; an SSH drop
should not kill them).

Cost note: an A5000 is about $0.27/hr. Stop the pod when you stop working.

---

## Track 1: measuring GPU work correctly

The single most common mistake in GPU benchmarking, and the reason the
T-VELCRO latency harness looks the way it does.

CUDA kernel launches are **asynchronous**. `t0 = time(); y = model(x); t1 = time()`
measures how long it took to *queue* the work, not to do it.

```python
import torch, time
x = torch.randn(4096, 4096, device="cuda")

# wrong
t0 = time.perf_counter(); y = x @ x; t1 = time.perf_counter()
print("wrong:", (t1 - t0) * 1e3, "ms")

# right
torch.cuda.synchronize()
t0 = time.perf_counter(); y = x @ x; torch.cuda.synchronize()
print("right:", (time.perf_counter() - t0) * 1e3, "ms")
```

Then: warm up (the first call includes autotuning and allocator growth), run
many iterations, take the **median** not the mean.

Read `common/timing.py` in this repo: it does exactly this, plus NVML energy
sampling. Understanding it is the point of this track.

**Exercises**
1. Measure a 4096x4096 matmul in fp32, tf32, and fp16. Explain the ratios from
   the A5000 spec sheet (fp32 vs tensor-core throughput).
2. Show that removing the warmup changes the reported number by more than 2x.
3. Compute achieved TFLOP/s and compare to the card's peak. What fraction do
   you reach, and why not 100 percent?

---

## Track 2: the memory wall

The concept that explains most GPU performance mysteries, including the one in
the user's paper.

**Arithmetic intensity** = FLOPs performed / bytes moved. Every kernel is
either compute-bound (high intensity) or memory-bound (low intensity). A
roofline plot makes this visible: below a threshold intensity, you are limited
by bandwidth and extra FLOPs are free; above it, you are limited by math.

```python
import torch, time

def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)

mm   = bench(lambda: a @ b)                 # compute bound: 2*n^3 FLOPs
add  = bench(lambda: a + b)                 # memory bound: 3*n^2*2 bytes
print(f"matmul {mm:.2f} ms  ->  {2*n**3/(mm*1e-3)/1e12:.1f} TFLOP/s")
print(f"add    {add:.2f} ms  ->  {3*n*n*2/(add*1e-3)/1e9:.0f} GB/s")
```

**Why this matters for T-VELCRO.** Removing 25 percent of attention heads
removes 25 percent of that layer's FLOPs, but the wall-clock saving is smaller,
because:
- the layer may be partly memory-bound (weights and activations still stream),
- GEMM tiles quantize: shrinking a dimension from 16 heads to 12 may not reduce
  the number of tiles proportionally,
- kernel launch overhead and non-attention layers are unchanged (Amdahl).

**Exercises**
1. Time `a @ b` with the inner dimension reduced by 25 percent. Is the speedup
   25 percent? Explain the gap.
2. Sweep matmul sizes from 512 to 8192 and plot achieved TFLOP/s. Find the size
   where the GPU stops being launch-latency-dominated.
3. Take `sam/surgery.py`, slice one block to 12 of 16 heads, and measure the
   real speedup of that block. Compare to the analytical 25 percent.

---

## Track 3: writing an actual CUDA kernel

```python
# save as add_kernel.py, run with: python add_kernel.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
__global__ void add_scaled(const float* a, const float* b, float* out,
                           float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s * b[i];
}
torch::Tensor add_scaled_cuda(torch::Tensor a, torch::Tensor b, double s) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_scaled<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                    out.data_ptr<float>(), (float)s, n);
    return out;
}
'''
mod = load_inline(name="myk", cpp_sources="torch::Tensor add_scaled_cuda(torch::Tensor, torch::Tensor, double);",
                  cuda_sources=cuda_src, functions=["add_scaled_cuda"], with_cuda=True)

a = torch.randn(1 << 20, device="cuda"); b = torch.randn_like(a)
torch.testing.assert_close(mod.add_scaled_cuda(a, b, 2.0), a + 2.0 * b)
print("kernel OK")
```

Concepts to internalize while doing this: thread/block/grid mapping, the
guard `if (i < n)`, coalesced access (consecutive threads touch consecutive
addresses), and why this kernel can never beat `a + 2*b` (both are bandwidth
bound and PyTorch's is already optimal).

**Exercises**
1. Write a kernel that sums each row of a matrix. Compare to `x.sum(dim=1)`.
   You will lose; find out by how much and why (hint: reduction strategy).
2. Make access non-coalesced on purpose (stride by 32) and measure the penalty.
3. Profile with `torch.profiler` and read the kernel timeline:
   ```python
   from torch.profiler import profile, ProfilerActivity
   with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p:
       mod.add_scaled_cuda(a, b, 2.0)
   print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   ```

---

## Track 4: attention, and why FlashAttention exists

Implement attention naively, then see the memory problem yourself.

```python
import torch, math
B, H, S, D = 1, 16, 4096, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)

def naive(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])   # B,H,S,S materialized
    return s.softmax(dim=-1) @ v

print("S x S matrix bytes:", B*H*S*S*2 / 1e9, "GB")
out_naive = naive(q, k, v)
out_flash = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print("max diff:", (out_naive - out_flash).abs().max().item())
```

The naive version materializes a B x H x S x S tensor: at S=4096, H=16 that is
about 2 GB in fp16, and it must be written to and read from HBM. FlashAttention
never materializes it, tiling the computation and keeping partial softmax
statistics in SRAM. Same FLOPs, far less memory traffic, several times faster.

This is the cleanest example of the Track 2 lesson: **the same arithmetic can be
fast or slow depending entirely on data movement.**

**Exercises**
1. Benchmark both at S = 1024, 2048, 4096. How does the gap scale?
2. At what S does the naive version OOM on a 24 GB card? Predict it first.
3. SAM ViT-H uses windowed attention (S=196) in 28 blocks and global (S=4096) in
   4. Compute the attention FLOPs and the S x S memory for each, and explain why
   the global blocks dominate cost. Compare with `sam/sam_velcro_pkg/flops.py`.

---

## Track 5: vLLM

Start with the server, then look inside.

```bash
tmux new -s vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096
# other shell:
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Explain paged attention:","max_tokens":100}'
```

Python API, which is better for experiments:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", max_model_len=4096,
          gpu_memory_utilization=0.85)
prompts = ["Write one sentence about GPUs."] * 64
out = llm.generate(prompts, SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

**The three ideas that make vLLM fast**

1. **KV cache**: each generated token attends to all previous ones, so keys and
   values are cached instead of recomputed. Cache size grows linearly with
   sequence length and batch size, and it is often the real memory bottleneck.
   Compute it by hand for your model: `2 * layers * kv_heads * head_dim * seq *
   batch * dtype_bytes`.
2. **PagedAttention**: the KV cache is stored in fixed-size blocks, like OS
   virtual memory pages, instead of one contiguous buffer per sequence. This
   removes fragmentation, so many more sequences fit in the same VRAM.
3. **Continuous batching**: finished sequences leave the batch and new requests
   join mid-flight, instead of waiting for the whole batch to finish. This is
   where most of the throughput gain comes from in serving.

**Exercises**
1. Compute the KV cache size for Qwen2.5-1.5B at 4096 tokens, batch 32. Then
   vary `gpu_memory_utilization` and find where vLLM starts refusing requests.
2. Measure throughput (tokens/s) at batch 1, 8, 64. Where does it saturate, and
   is that limit compute or memory bandwidth?
3. Run the official benchmark and read the scheduler stats:
   `python -m vllm.entrypoints.openai.api_server --help` and the metrics at
   `http://localhost:8000/metrics`.
4. Load a 7B model in fp16 on 24 GB. Does it fit? Compute weights plus KV cache
   before trying, then check your prediction.

---

## Track 6: bringing it back to T-VELCRO

Questions the user's own paper leaves open, which this knowledge answers:

1. **Why is measured speedup below the analytical FLOP reduction?** Decompose
   it: what fraction of encoder time is attention vs MLP vs everything else?
   Use `torch.profiler` on the SAM encoder and attribute time per module.
2. **Do windowed and global blocks behave differently under head removal?**
   Predict from arithmetic intensity (S=196 vs S=4096), then measure.
3. **Is the hook implementation slower than baseline, and by exactly how much?**
   It computes the head and then overwrites it, so it should be, and the
   benchmark reports it. Quantify the overhead per bypassed head.
4. **Would fusing the constant term help?** In `sam/surgery.py` the frozen
   heads' contribution is added after the projection. Could it fold into the
   bias instead (as `sam/surgery_mlp.py` does for MLP neurons), and would that
   measurably matter?

Answering 1 and 2 with real profiles would strengthen the paper's discussion
section beyond what any reviewer asked for.

---

## Reference

- PyTorch CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- CUDA C++ programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- FlashAttention paper: https://arxiv.org/abs/2205.14135
- PagedAttention / vLLM paper: https://arxiv.org/abs/2309.06180
- vLLM docs: https://docs.vllm.ai/

In-repo code worth reading as examples:
- `common/timing.py` - correct GPU timing and energy measurement
- `sam/surgery.py`, `vit/surgery_dinov3.py` - structured weight removal
- `sam/sam_velcro_pkg/flops.py` - analytical cost model to compare against
