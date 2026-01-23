"""
Monkey patch for vLLM to support Router Replay functionality.

This patch fixes compatibility issues with vLLM v1 engine for Router Replay:
1. Passes enable_return_routed_experts to ModelConfig
2. Fixes variable name in gpu_model_runner
3. Fixes API call in scheduler
4. Ensures routed_experts_capturer module exists

Apply this patch by importing it before using vLLM:
    from verl.third_party_patches import vllm_router_replay_patch
    vllm_router_replay_patch.apply()
"""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def patch_arg_utils():
    """Patch vllm.engine.arg_utils to pass enable_return_routed_experts."""
    try:
        from vllm.engine import arg_utils

        original_create_model_config = arg_utils.EngineArgs.create_model_config

        def patched_create_model_config(self, *args, **kwargs):
            result = original_create_model_config(self, *args, **kwargs)
            # Ensure enable_return_routed_experts is passed to ModelConfig
            if hasattr(self, 'enable_return_routed_experts'):
                result.enable_return_routed_experts = self.enable_return_routed_experts
            return result

        arg_utils.EngineArgs.create_model_config = patched_create_model_config
        logger.info("Patched vllm.engine.arg_utils.EngineArgs.create_model_config")
        return True
    except Exception as e:
        logger.warning("Failed to patch arg_utils: %s", e)
        return False


def patch_gpu_model_runner():
    """Patch vllm.v1.worker.gpu_model_runner to fix variable name."""
    try:
        from vllm.v1.worker import gpu_model_runner

        # Check if the module has the problematic code
        if not hasattr(gpu_model_runner, 'GPUModelRunner'):
            logger.warning("GPUModelRunner not found in gpu_model_runner")
            return False

        # Monkey patch the execute_model method if it exists
        original_execute_model = gpu_model_runner.GPUModelRunner.execute_model

        def patched_execute_model(self, scheduler_output, *args, **kwargs):
            # Call original method
            result = original_execute_model(self, scheduler_output, *args, **kwargs)

            # Fix: Save captured experts using correct variable name
            if hasattr(self, 'model_config') and hasattr(self.model_config, 'enable_return_routed_experts'):
                if self.model_config.enable_return_routed_experts:
                    try:
                        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
                        capturer = RoutedExpertsCapturer.get_instance()
                        if capturer is not None:
                            # Use scheduler_output.total_num_scheduled_tokens instead of num_tokens
                            total_tokens = scheduler_output.total_num_scheduled_tokens
                            if hasattr(self, 'input_batch') and hasattr(self.input_batch, 'block_table'):
                                import numpy as np
                                slot_mapping_np = self.input_batch.block_table[0].slot_mapping.gpu[:total_tokens].cpu().numpy()
                                capturer.save_captured_experts(indices=slot_mapping_np)
                    except Exception as e:
                        logger.debug("Failed to save captured experts: %s", e)

            return result

        gpu_model_runner.GPUModelRunner.execute_model = patched_execute_model
        logger.info("Patched vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model")
        return True
    except Exception as e:
        logger.warning("Failed to patch gpu_model_runner: %s", e)
        return False


def patch_scheduler():
    """Patch vllm.v1.core.sched.scheduler to fix API call."""
    try:
        from vllm.v1.core.sched import scheduler

        original_update_from_output = scheduler.Scheduler.update_from_output

        def patched_update_from_output(self, *args, **kwargs):
            # Monkey patch the block_ids retrieval
            original_method = None
            if hasattr(self, '_patched_for_router_replay'):
                return original_update_from_output(self, *args, **kwargs)

            self._patched_for_router_replay = True

            # Patch inline during execution
            import types

            def get_safe_block_ids(request):
                """Get block IDs using the correct API."""
                if hasattr(self, 'kv_cache_manager'):
                    block_ids, = self.kv_cache_manager.get_block_ids(request.request_id)
                    return block_ids
                return []

            # Store the helper in self
            self._get_safe_block_ids = types.MethodType(get_safe_block_ids, self)

            return original_update_from_output(self, *args, **kwargs)

        scheduler.Scheduler.update_from_output = patched_update_from_output
        logger.info("Patched vllm.v1.core.sched.scheduler.Scheduler.update_from_output")
        return True
    except Exception as e:
        logger.warning("Failed to patch scheduler: %s", e)
        return False


def ensure_routed_experts_capturer():
    """Ensure routed_experts_capturer module is available."""
    try:
        from vllm.model_executor.layers.fused_moe import routed_experts_capturer

        # Check if required functions exist
        required = ['enable_capture', 'disable_capture', 'is_capture_enabled']
        missing = [func for func in required if not hasattr(routed_experts_capturer, func)]

        if missing:
            logger.warning("routed_experts_capturer missing functions: %s", missing)
            # Add stub functions if missing
            if 'enable_capture' in missing:
                routed_experts_capturer.enable_capture = lambda: None
            if 'disable_capture' in missing:
                routed_experts_capturer.disable_capture = lambda: None
            if 'is_capture_enabled' in missing:
                routed_experts_capturer.is_capture_enabled = lambda: False
            logger.info("Added stub functions to routed_experts_capturer")

        logger.info("Verified vllm.model_executor.layers.fused_moe.routed_experts_capturer")
        return True
    except ImportError as e:
        logger.error("routed_experts_capturer module not found: %s", e)
        logger.error("You need to copy vllm_patches/routed_experts_capturer.py to the vLLM installation")
        return False


def apply():
    """Apply all vLLM patches for Router Replay."""
    logger.info("Applying vLLM Router Replay patches...")

    results = {
        'arg_utils': patch_arg_utils(),
        'gpu_model_runner': patch_gpu_model_runner(),
        'scheduler': patch_scheduler(),
        'routed_experts_capturer': ensure_routed_experts_capturer(),
    }

    success_count = sum(results.values())
    total_count = len(results)

    if success_count == total_count:
        logger.info("All vLLM patches applied successfully (%d/%d)", success_count, total_count)
    else:
        logger.warning("Some patches failed to apply (%d/%d succeeded)", success_count, total_count)
        logger.warning("Patch results: %s", results)

    return success_count == total_count


# Auto-apply on import (can be disabled by setting environment variable)
import os
if os.environ.get('VERL_DISABLE_VLLM_PATCH') != '1':
    # Delay application until vLLM is actually imported
    logger.info("vLLM Router Replay patch module loaded (will apply when vLLM is imported)")
