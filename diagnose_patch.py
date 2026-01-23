"""Diagnose monkey patch issues."""
import sys
import os

print("=" * 60)
print("STEP 1: Check verl package installation")
print("=" * 60)
try:
    import verl
    print(f"✓ verl package found")
    print(f"  Location: {verl.__file__}")
    print(f"  Path: {os.path.dirname(verl.__file__)}")
except ImportError as e:
    print(f"✗ verl package not found: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 2: Check third_party_patches module")
print("=" * 60)
try:
    from verl import third_party_patches
    print(f"✓ third_party_patches module found")
    print(f"  Location: {third_party_patches.__file__}")
except ImportError as e:
    print(f"✗ third_party_patches module not found: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 3: Check vllm_router_replay_patch module")
print("=" * 60)
try:
    from verl.third_party_patches import vllm_router_replay_patch
    print(f"✓ vllm_router_replay_patch module found")
    print(f"  Location: {vllm_router_replay_patch.__file__}")
    print(f"  Has apply(): {hasattr(vllm_router_replay_patch, 'apply')}")
    print(f"  Has patch_arg_utils(): {hasattr(vllm_router_replay_patch, 'patch_arg_utils')}")
except ImportError as e:
    print(f"✗ vllm_router_replay_patch module not found: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 4: Apply patch with detailed logging")
print("=" * 60)
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('verl.third_party_patches.vllm_router_replay_patch')
logger.setLevel(logging.DEBUG)

result = vllm_router_replay_patch.apply()
print(f"\nPatch apply() result: {result}")

print("\n" + "=" * 60)
print("STEP 5: Check EngineArgs class attribute")
print("=" * 60)
try:
    from vllm.engine import arg_utils
    print(f"✓ vllm.engine.arg_utils imported")

    # Check if EngineArgs exists
    if hasattr(arg_utils, 'EngineArgs'):
        print(f"✓ EngineArgs class found")

        # Check dataclass
        import dataclasses
        is_dc = dataclasses.is_dataclass(arg_utils.EngineArgs)
        print(f"  Is dataclass: {is_dc}")

        # Check attribute
        has_attr = hasattr(arg_utils.EngineArgs, 'enable_return_routed_experts')
        print(f"  Has enable_return_routed_experts: {has_attr}")

        if has_attr:
            value = getattr(arg_utils.EngineArgs, 'enable_return_routed_experts')
            print(f"  Default value: {value}")
            print(f"  Type: {type(value)}")

            # Check annotation
            if hasattr(arg_utils.EngineArgs, '__annotations__'):
                annot = arg_utils.EngineArgs.__annotations__.get('enable_return_routed_experts', 'NOT FOUND')
                print(f"  Type annotation: {annot}")
        else:
            print("\n✗ PROBLEM: enable_return_routed_experts attribute NOT FOUND")
            print("\n  Debugging info:")
            print(f"  EngineArgs.__dict__ keys: {list(arg_utils.EngineArgs.__dict__.keys())[:10]}...")
            if hasattr(arg_utils.EngineArgs, '__annotations__'):
                print(f"  EngineArgs.__annotations__ keys: {list(arg_utils.EngineArgs.__annotations__.keys())[:10]}...")
    else:
        print("✗ EngineArgs class not found")
except Exception as e:
    print(f"✗ Error checking EngineArgs: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 6: Check if patch functions were called")
print("=" * 60)
# Try to manually call patch_arg_utils
try:
    print("Attempting to manually call patch_arg_utils()...")
    result = vllm_router_replay_patch.patch_arg_utils()
    print(f"patch_arg_utils() returned: {result}")

    # Check again
    from vllm.engine import arg_utils
    has_attr_after = hasattr(arg_utils.EngineArgs, 'enable_return_routed_experts')
    print(f"After manual patch, has attribute: {has_attr_after}")
except Exception as e:
    print(f"Error calling patch_arg_utils(): {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
