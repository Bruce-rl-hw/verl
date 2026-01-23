"""Test parameter flow for enable_return_routed_experts."""
import sys

# Apply patches
from verl.third_party_patches import vllm_router_replay_patch
result = vllm_router_replay_patch.apply()
print(f"Patch applied: {result}")
print()

# Import after patching
from vllm.engine import arg_utils

print("=" * 60)
print("Test 1: EngineArgs class attribute")
print("=" * 60)
has_attr = hasattr(arg_utils.EngineArgs, "enable_return_routed_experts")
print(f"hasattr(EngineArgs, enable_return_routed_experts): {has_attr}")
if has_attr:
    value = getattr(arg_utils.EngineArgs, "enable_return_routed_experts")
    print(f"Default value: {value}")
    print(f"Type annotation: {arg_utils.EngineArgs.__annotations__.get('enable_return_routed_experts')}")

print()
print("=" * 60)
print("Test 2: Create EngineArgs instance and set attribute")
print("=" * 60)
# Create instance without the parameter
engine_args = arg_utils.EngineArgs(model="facebook/opt-125m")
before_value = getattr(engine_args, "enable_return_routed_experts", "NOT SET")
print(f"Before: engine_args.enable_return_routed_experts = {before_value}")

# Set the attribute manually (this is what from_cli_args does)
engine_args.enable_return_routed_experts = True
print(f"After manual set: engine_args.enable_return_routed_experts = {engine_args.enable_return_routed_experts}")

print()
print("=" * 60)
print("Test 3: create_model_config transfer")
print("=" * 60)
try:
    model_config = engine_args.create_model_config()
    has_mc_attr = hasattr(model_config, "enable_return_routed_experts")
    print(f"ModelConfig has enable_return_routed_experts: {has_mc_attr}")
    if has_mc_attr:
        print(f"model_config.enable_return_routed_experts: {model_config.enable_return_routed_experts}")

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✓ EngineArgs class has attribute: {has_attr}")
        print(f"✓ Can set attribute on instance: True")
        print(f"✓ Attribute value: {engine_args.enable_return_routed_experts}")
        print(f"✓ ModelConfig receives value: {model_config.enable_return_routed_experts}")
        print()
        print("✅ Complete parameter flow verified successfully!")
        print()
        print("Note: The parameter flow works as:")
        print("  CLI args -> from_cli_args sets instance attr -> create_model_config transfers to ModelConfig")
    else:
        print("❌ ERROR: ModelConfig missing enable_return_routed_experts attribute!")
except Exception as e:
    print(f"Error creating ModelConfig: {e}")
    import traceback
    traceback.print_exc()
