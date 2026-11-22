import os
import signal
import subprocess

os.environ["USE_LIBUV"] = "0"

import torch.distributed as dist

_orig_store = dist.TCPStore


def _patched_store(*args, **kwargs):
    kwargs["use_libuv"] = False
    return _orig_store(*args, **kwargs)


dist.TCPStore = _patched_store

_orig_send = subprocess.Popen.send_signal


def _send(self, sig):
    if sig == getattr(signal, "CTRL_C_EVENT", None):
        self.terminate()
    else:
        _orig_send(self, sig)


subprocess.Popen.send_signal = _send

from torch.distributed.run import main

if __name__ == "__main__":
    main()