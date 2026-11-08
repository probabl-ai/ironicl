"""Progress for ``verbose=True``: a tqdm bar when tqdm is installed, else lines on stderr."""

from __future__ import annotations

import sys

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional: pip install ironicl[progress]
    tqdm = None


class Progress:
    """Shows ``done`` of ``total`` ``unit``\\s for one task as it advances."""

    def __init__(self, description: str, unit: str, total: int | None = None):
        self.description = description
        self.unit = unit
        self.total = total
        self._bar = None
        self._tenths = -1

    def update(self, done: int, total: int | None = None) -> None:
        total = self.total if self.total is not None else total
        if tqdm is not None:
            if self._bar is None:
                self._bar = tqdm(total=total, desc=self.description, unit=self.unit)
            self._bar.n = done
            self._bar.refresh()
            return
        # Without tqdm, a line per tenth of the way.
        tenths = done * 10 // max(total or 1, 1)
        if tenths > self._tenths:
            self._tenths = tenths
            print(f"ironicl: {self.description}: {done}/{total} {self.unit}s", file=sys.stderr, flush=True)

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, *_) -> None:
        if self._bar is not None:
            self._bar.close()
