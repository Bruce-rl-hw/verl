# MegatronAdaptor + TransformerEngine-NPU Backend

Last updated: 07/24/2026.

`megatron_adaptor` is a Megatron training-engine backend for Ascend NPU, provided as an
alternative to the MindSpeed path. It runs the lightweight **MegatronAdaptor** patch layer
together with **TransformerEngine-NPU (TE-NPU)**, which lets verl track newer `megatron-core`
releases and unlocks **mxfp8** (MXFP8 block scaling) low-precision training on NPU.

It reuses the generic Megatron engine (`MegatronEngineWithLMHead` / `MegatronEngineWithValueHead`);
all NPU patches are applied once at `import megatron_adaptor`. It coexists with the `megatron`
and `mindspeed` backends and does not change their behavior.

## Requirements

Install the following on your Ascend environment (they are not pulled in automatically):

| Component | Note |
| --- | --- |
| `megatron_adaptor` | Ascend MegatronAdaptor patch layer |
| `transformer_engine` | TransformerEngine-NPU (drop-in replacement) |
| `megatron-core` | the version supported by your MegatronAdaptor release |

## Usage

Select the backend through the engine config group (or by overriding the strategy directly):

```bash
python3 -m verl.trainer.main_ppo \
    engine=megatron_adaptor \
    actor_rollout_ref.actor.strategy=megatron_adaptor \
    ...
```

To enable mxfp8 low-precision training, pass the fp8 recipe through
`override_transformer_config`:

```bash
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=mxfp8
```

All other Megatron parallel/offload options
(`tensor_model_parallel_size`, `expert_model_parallel_size`, `param_offload`, ...) behave the
same as the `megatron` backend; see `verl/trainer/config/engine/megatron_adaptor.yaml`.

## Notes

- `strategy=megatron_adaptor` is accepted by `McoreEngineConfig` alongside `megatron`.
- fp8 is forwarded only when explicitly set via `override_transformer_config`; otherwise the
  historical behavior (fp8 disabled) is preserved, so existing users are unaffected.
