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

import pytest

try:
    from verl.workers.engine import MegatronAdaptorEngineWithLMHead as _AdaptorLMEngine
except Exception:
    _AdaptorLMEngine = None

# The megatron_adaptor engine classes register at import of verl.workers.engine. Registration
# only needs megatron-core (not torch_npu / the megatron_adaptor patch layer), so it succeeds on
# the GPU-based CI image and the registry assertions below run there. When megatron-core itself is
# absent the classes are not importable, so those assertions are skipped instead of failing.
_ADAPTOR_UNAVAILABLE = _AdaptorLMEngine is None
_ADAPTOR_SKIP_REASON = "megatron / megatron_adaptor engine not importable in this environment"


def test_mcore_config_accepts_adaptor_strategy():
    from verl.workers.config import McoreEngineConfig

    cfg = McoreEngineConfig(strategy="megatron_adaptor", tensor_model_parallel_size=2)
    assert cfg.strategy == "megatron_adaptor"


def test_mcore_config_rejects_unknown_strategy():
    from verl.workers.config import McoreEngineConfig

    with pytest.raises(AssertionError):
        McoreEngineConfig(strategy="not_a_backend")


@pytest.mark.skipif(_ADAPTOR_UNAVAILABLE, reason=_ADAPTOR_SKIP_REASON)
def test_adaptor_engine_registered_under_npu_backend():
    from verl.workers.engine.base import EngineRegistry

    lm = EngineRegistry._engines["language_model"]
    vm = EngineRegistry._engines["value_model"]

    assert lm["megatron_adaptor"]["npu"].__name__ == "MegatronAdaptorEngineWithLMHead"
    assert vm["megatron_adaptor"]["npu"].__name__ == "MegatronAdaptorEngineWithValueHead"


@pytest.mark.skipif(_ADAPTOR_UNAVAILABLE, reason=_ADAPTOR_SKIP_REASON)
def test_adaptor_does_not_shadow_plain_megatron_backend():
    from verl.workers.engine.base import EngineRegistry

    lm = EngineRegistry._engines["language_model"]
    # megatron_adaptor must be a distinct backend entry, never overwriting the plain
    # megatron backend's registrations.
    for engine_cls in lm.get("megatron", {}).values():
        assert engine_cls.__name__ != "MegatronAdaptorEngineWithLMHead"
