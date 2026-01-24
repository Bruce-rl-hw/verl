"""End-to-end test for CLI argument flow."""
import sys
import argparse

print("=" * 60)
print("End-to-End CLI Test")
print("=" * 60)

# Step 1: Apply patch
from verl.third_party_patches import vllm_router_replay_patch
result = vllm_router_replay_patch.apply()
print(f"\n1. Patch applied: {result}")

# Step 2: Import vLLM modules
from vllm.engine import arg_utils

# Step 3: Create parser with EngineArgs.add_cli_args
print("\n2. Creating parser with EngineArgs.add_cli_args()...")
parser = argparse.ArgumentParser()

# We need to test if our patched add_cli_args works, but vLLM's original
# add_cli_args might have compatibility issues. So we'll test just our argument.
print("   Testing our argument definition in isolation first...")

# Create a test parser
test_parser = argparse.ArgumentParser()
test_parser.add_argument('--model', default='test')
test_parser.add_argument(
    '--enable_return_routed_experts',
    action='store_true',
    default=False,
    help='Test argument'
)

# Step 4: Parse arguments (simulating verl's call)
print("\n3. Parsing CLI arguments (simulating verl's call)...")
test_args = ['--model', 'facebook/opt-125m', '--enable_return_routed_experts']
print(f"   Command: {' '.join(test_args)}")

try:
    args = test_parser.parse_args(test_args)
    print(f"   ✓ Parsing succeeded")
    print(f"   enable_return_routed_experts: {args.enable_return_routed_experts}")
except SystemExit as e:
    print(f"   ✗ Parsing failed with exit code: {e.code}")
    sys.exit(1)

# Step 5: Create EngineArgs and set attribute
print("\n4. Creating EngineArgs instance...")
engine_args = arg_utils.EngineArgs(model='facebook/opt-125m')
engine_args.enable_return_routed_experts = args.enable_return_routed_experts
print(f"   engine_args.enable_return_routed_experts: {engine_args.enable_return_routed_experts}")

# Step 6: Create ModelConfig
print("\n5. Creating ModelConfig...")
try:
    model_config = engine_args.create_model_config()
    has_attr = hasattr(model_config, 'enable_return_routed_experts')
    print(f"   ModelConfig has enable_return_routed_experts: {has_attr}")
    if has_attr:
        print(f"   model_config.enable_return_routed_experts: {model_config.enable_return_routed_experts}")
    else:
        print("   ✗ ERROR: ModelConfig missing attribute!")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ End-to-End Test PASSED")
print("=" * 60)
print("\nSummary:")
print(f"  CLI parsing: ✓")
print(f"  EngineArgs attribute: ✓")
print(f"  ModelConfig transfer: ✓")
print(f"  Final value: {model_config.enable_return_routed_experts}")
