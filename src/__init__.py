from contextlib import contextmanager
import torch

# Root context is cpu device
_context_stack = [torch.device("cpu")]


@contextmanager
def device(name: str):
    """Creates a device context to run in.

    Args:
        name: pytorch name of the device, e.g. "cpu" or "cuda:0"

    """
    _context_stack.append(torch.device(name))
    yield
    _context_stack.pop()


def get_device() -> torch.device:
    """Gets the current torch device.

    Returns: the torch device that is currently active

    """
    return _context_stack[-1]
