"""Verify actual CPU/GPU aliasing, ownership and repeated sampler updates."""
import gc
import numpy as np
import pytest
import torch
from glm53_uva_view import mapped_view


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.float32])
def test_cpu_updates_visible_and_owner_retained(dtype):
    x = torch.zeros(16, dtype=dtype, pin_memory=True)
    view = mapped_view(x)
    for value in [37, 50, 1, 99]:
        x.fill_(value)
        assert view.cpu().tolist() == [value]*16
    del x
    gc.collect()
    assert view.cpu().tolist() == [99]*16


def test_sampler_parameters_survive_pool_rotation():
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.buffer_utils import set_default_max_concurrency
    from vllm.sampling_params import SamplingParams
    set_default_max_concurrency(3)
    state = SamplingStates(16, 154880)
    for i in range(16):
        state.add_request(i, SamplingParams.for_sampler_warmup())
    for _ in range(9):
        state.apply_staged_writes()
        k,p = state.get_top_k_top_p(torch.tensor([15,15,14],device='cuda'),np.array([15,14]))
        assert k.cpu().tolist() == [50]*3
        assert p.cpu().tolist() == pytest.approx([0.9]*3)
