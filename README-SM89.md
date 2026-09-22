# NInfer Qwen3.8 27B on RTX 4060 Ti 16 GB

This local SM89 port runs the Qwen3.8 27B NInfer groupwise-int artifact with a **65,536-token logical context and 65,536-token physical Q4 KV cache**. It serves an OpenAI-compatible API on `127.0.0.1:18088`, with MTP-3 and vision enabled. It uses one GPU and one concurrent request.

## Start, inspect, stop

From PowerShell:

```powershell
& 'C:\Users\barrett\ninfer-sm89\scripts\ninfer-sm89.ps1' Start
& 'C:\Users\barrett\ninfer-sm89\scripts\ninfer-sm89.ps1' Status
& 'C:\Users\barrett\ninfer-sm89\scripts\ninfer-sm89.ps1' Stop
```

Startup loads 16.46 GB of model weights from `E:\modelmadness\ninfer-5080\qwen3_8_27b.ninfer` and can take about two minutes. `Start` reuses its own running container. `Stop` only stops `ninfer-sm89-64k`. The API base is `http://127.0.0.1:18088/v1`, model ID `qwen3.8-27b-sm89`. It is bound to loopback on Windows; there is no API authentication.

```powershell
curl.exe -s http://127.0.0.1:18088/v1/models
```

For an OpenAI-compatible coding client, set base URL to `http://127.0.0.1:18088/v1` and model to `qwen3.8-27b-sm89`. If the client insists on an API key, use a dummy local value; the server does not require one. Keep client concurrency at one. The server's chat completions route supports tools and image messages.

## Rebuild

The port is based on upstream NInfer commit `20296062f050dff6e3e688fa36934efd1330d9c8` (tag `qwen3.8-27b-rtx5080-128k-vision-v1.3`). The toolchain image uses CUDA 13.1.2 on Ubuntu 24.04. The machine tested has NVIDIA driver 591.86 and RTX 4060 Ti (SM89, 16,380 MiB).

```powershell
Set-Location C:\Users\barrett\ninfer-sm89
docker build -t ninfer-sm89-toolchain -f docker/ada-toolchain.Dockerfile .
docker run --rm --mount type=bind,source=C:\Users\barrett\ninfer-sm89,target=/src --workdir /src ninfer-sm89-toolchain cmake -S . -B build-sm89 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON
docker run --rm --mount type=bind,source=C:\Users\barrett\ninfer-sm89,target=/src --workdir /src ninfer-sm89-toolchain cmake --build build-sm89 --parallel 6 --target ninfer-serve
```

The launcher uses this container profile:

```text
ninfer-serve /model/qwen3_8_27b.ninfer --host 0.0.0.0 --port 8080
  --model-id qwen3.8-27b-sm89 --max-context 65536 --kv-capacity 65536
  --prefill-chunk 896 --kv-dtype q4 --spec mtp --draft-tokens 3
  --no-cuda-graph --max-concurrency 1 --default-thinking-budget 2048
  --prefix-checkpoint-policy rolling-tool --vision --vision-max-tokens 1792
```

The artifact SHA-256 is `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`. Docker image ID used in the first successful run: `sha256:c7d7e0101ae3ec4c9c13cf9296bb572836b933733f25c68c9b4620b8580cc5f5`.

## Measured on this machine

- Startup allocated the full 65,536-token Q4 KV capacity and reported about **780 MiB CUDA free after startup**. NVIDIA's external free-memory count was about 1.1 GiB. These counters differ by allocator and timing.
- A fresh 60,092-token prompt recovered a six-digit code placed at its beginning. Server log: **100,331 ms TTFT**, **599.8 prefill tok/s**, **29.1 decode tok/s**, 25 generated tokens, MTP 3.67 tokens per round (88.9%) for this one retrieval prompt. A second near-edge prompt also retrieved the correct code with **65,452 prompt tokens and 25 completion tokens**.
- A small chat prompt returned `pong`. Five required `write_code` calls returned valid `path` and `code` arguments, including Python and TypeScript requests. A 512×256 red/blue image was identified correctly.
- The SM89 `ninfer_gdn_gating_proj_test` GPU regression passes. Its Ada FP32 tolerance accounts for the unsplit accumulation order; Blackwell test tolerances remain unchanged.

## Limitations

- **Model scope:** The validated path is the Qwen3.8-27B `groupwise-int` artifact. This is not a validation of all registered NInfer identities. The 35B-A3B/DFlash profile has not been qualified on SM89.
- **NVFP4:** The loader rejects NVFP4 artifacts on SM89. Their W4A4/TMA kernels require SM120, and the corresponding SM89 routes fail explicitly.
- **Other W8 routes:** Several W8 split-K operator routes for other profiles exceed Ada's static shared-memory limit or were not ported. They are excluded from the SM89 build and replaced by explicit errors. DFlash has not been qualified on SM89. This does not disable the tested Qwen3.8 MTP-3 path.
- **GDN fallback:** Blackwell-tuned cooperative GDN projection routes are disabled on SM89. The unsplit MMA route passes the GPU regression test with an Ada-specific FP32 tolerance, but may have different performance and rounding.
- **Memory and context:** CUDA graphs are disabled in the launcher to preserve VRAM headroom; graph operation has not been qualified. The 65,536-token physical Q4 KV allocation and a 65,452-token prompt plus 25 generated tokens succeeded. A request at the exact 65,536-token boundary and 128K on this 16 GB card were not validated.
- **Vision and agent quality:** One image was tested; video input was not. Chat, five required tool calls, image identification, and long-context retrieval are smoke checks, not an agent-quality or stability benchmark. MTP acceptance and throughput vary by prompt.
