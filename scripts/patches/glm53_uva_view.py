"""Narrow compatibility shim for this deployment's false-negative is_pinned()."""
from pathlib import Path
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    cuda = Path(os.environ['CUDA_HOME'])
    return load(name='glm53_uva_view_ext',
                sources=[str(Path(__file__).with_suffix('.cpp'))],
                extra_include_paths=[str(cuda/'include')],
                extra_cflags=['-O2'],
                extra_ldflags=['-L'+str(cuda/'lib'), '-lcudart'],
                with_cuda=False, verbose=False)


def mapped_view(tensor):
    return extension().mapped_view(tensor)
