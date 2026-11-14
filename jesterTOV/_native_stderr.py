"""Filter one repetitive XLA native warning during command-line inference."""

from contextlib import contextmanager
import os
import re
import sys
import threading
from collections.abc import Iterator

# XLA emits this from C++ directly to fd 2, outside Python's logging system.
_SOL_WARNING = re.compile(
    rb"^W\d{4} .*sol_gpu_cost_model\.cc:\d+\] "
    rb"No SoL config found for device: .*\. Using default config\.\s*$"
)


@contextmanager
def filter_xla_cost_model_warning() -> Iterator[None]:
    """Suppress the known missing-SoL warning while preserving other stderr.

    Scope the file-descriptor redirect to CLI execution. The reader owns a
    duplicate of the original descriptor, so other native warnings and errors
    retain their destination even if stderr was redirected by the caller.
    """
    saved = os.dup(2)
    output = os.dup(saved)
    read_fd, write_fd = os.pipe()

    def forward() -> None:
        try:
            with os.fdopen(read_fd, "rb") as stream:
                for line in stream:
                    if not _SOL_WARNING.match(line):
                        remaining = memoryview(line)
                        while remaining:
                            remaining = remaining[os.write(output, remaining) :]
        finally:
            os.close(output)

    reader = threading.Thread(target=forward, name="jester-stderr-filter", daemon=True)
    reader.start()
    try:
        sys.stderr.flush()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
    finally:
        os.close(saved)
        reader.join(timeout=5)
