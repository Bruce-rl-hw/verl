# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ADAPTOR_REQUIRED_MSG = (
    "backend='megatron_adaptor' requires the megatron_adaptor package "
    "(https://gitcode.com/Ascend/MegatronAdaptor) to be installed"
)


def _apply_megatron_adaptor_patch():
    """Import megatron_adaptor to apply the Ascend NPU monkey patches to megatron.core.

    Done lazily (mirroring the MindSpeed backend's ``repatch()``) so the patches are applied only
    when a megatron_adaptor engine is actually initialized -- importing this module for engine
    registration, or selecting another backend, never triggers them. This also avoids clashing
    with the top-level ``megatron_adaptor`` module shipped by MindSpeed-LLM on Ascend images.
    """
    try:
        import megatron_adaptor  # noqa: F401
    except Exception as e:
        # torch_npu raises non-ImportError (e.g. UnicodeDecodeError) when the CANN env is missing.
        raise AssertionError(_ADAPTOR_REQUIRED_MSG) from e


@EngineRegistry.register(model_type="language_model", backend="megatron_adaptor", device="npu")
class MegatronAdaptorEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron engine on Ascend NPU via the lightweight MegatronAdaptor patch layer."""

    def _init_device_mesh(self):
        # Apply the MegatronAdaptor patches before initialize_model_parallel, mirroring the
        # MindSpeed backend, so the patched megatron.core APIs are in effect at model build time.
        _apply_megatron_adaptor_patch()
        super()._init_device_mesh()


@EngineRegistry.register(model_type="value_model", backend="megatron_adaptor", device="npu")
class MegatronAdaptorEngineWithValueHead(MegatronEngineWithValueHead):
    """Value-model variant of the MegatronAdaptor NPU engine."""

    def _init_device_mesh(self):
        _apply_megatron_adaptor_patch()
        super()._init_device_mesh()
