import os

os.environ["USE_LIBUV"] = "0"

import torch.distributed as dist

_orig = dist.TCPStore


def _patched(*args, **kwargs):
    kwargs["use_libuv"] = False
    return _orig(*args, **kwargs)


dist.TCPStore = _patched

from torch.distributed.run import main

if __name__ == "__main__":
    main()