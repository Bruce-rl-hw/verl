# Third-Party Patches

This directory contains monkey patches for third-party libraries to fix compatibility issues.

## vLLM Router Replay Patch

### What it does

Fixes vLLM v1 engine compatibility for Router Replay (R3 mode):

1. **arg_utils.py**: Passes `enable_return_routed_experts` to ModelConfig
2. **gpu_model_runner.py**: Fixes `num_tokens` → `scheduler_output.total_num_scheduled_tokens`
3. **scheduler.py**: Fixes `request.get_cached_block_ids()` → `self.kv_cache_manager.get_block_ids()`
4. **routed_experts_capturer.py**: Ensures module exists with required functions

### How to use

#### Automatic (Recommended)

The patch is **automatically applied** when you import `vllm_async_server.py`. No manual action needed.

#### Manual Application

If you need to apply patches in other code:

```python
from verl.third_party_patches import vllm_router_replay_patch
vllm_router_replay_patch.apply()
```

#### Disable Auto-patch

Set environment variable before running:

```bash
export VERL_DISABLE_VLLM_PATCH=1
```

### Important: routed_experts_capturer.py

**The `routed_experts_capturer.py` module MUST be manually installed** as it's a complete new file (not a monkey patch).

On new machines:

```bash
# Copy the file to vLLM installation
docker cp vllm_patches/routed_experts_capturer.py <container>:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/
```

### Verification

Check logs for:
- ✅ `All vLLM patches applied successfully (4/4)`
- ⚠️ `Some patches failed to apply` - indicates missing dependencies

### Troubleshooting

If patches fail:
1. Check vLLM version compatibility
2. Ensure `routed_experts_capturer.py` is installed
3. Check logs for specific error messages
4. Fall back to manual file replacement (see `vllm_patches/` directory)
