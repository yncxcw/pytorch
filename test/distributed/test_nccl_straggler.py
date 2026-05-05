# Owner(s): ["oncall: distributed"]

import os
import sys
import time

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


# How long the slow rank should sleep before entering the collective in the
# "straggler" test. Picked to be safely above the threshold (500 ms) and the
# rank-0 poll cadence (50 ms).
STRAGGLER_SLEEP_S = 1.5


class NcclStragglerTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        # Each test sets its own env vars; we spawn here so children inherit.
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def device(self):
        return torch.device("cuda", self.rank % torch.cuda.device_count())

    def _capture_stderr_path(self):
        # Use a per-rank file in the same dir as self.file_name so the test
        # framework cleans the directory up.
        return f"{self.file_name}.stderr.{self.rank}"

    def _redirect_stderr_to_file(self, path):
        # Force C++ glog warnings (used by ProcessGroupNCCL) to be flushed to
        # stderr immediately, then dup2 stderr to a file we can read after
        # the process exits.
        os.environ["GLOG_logtostderr"] = "1"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        sys.stderr.flush()
        f = open(path, "w", buffering=1)
        os.dup2(f.fileno(), sys.stderr.fileno())
        # Keep f alive for the lifetime of the process by stashing on self.
        self._stderr_file = f

    def _init_pg(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_straggler_detected(self):
        os.environ["TORCH_NCCL_STRAGGLER_DETECT"] = "1"
        os.environ["TORCH_NCCL_STRAGGLER_THRESHOLD_MS"] = "500"
        os.environ["TORCH_NCCL_STRAGGLER_POLL_MS"] = "50"

        stderr_path = self._capture_stderr_path()
        self._redirect_stderr_to_file(stderr_path)

        self._init_pg()
        device = self.device
        torch.cuda.set_device(device)

        # Rank 1 is the straggler: enters the collective much later.
        if self.rank == 1:
            time.sleep(STRAGGLER_SLEEP_S)

        x = torch.ones(4, device=device)
        c10d.all_reduce(x)
        torch.cuda.synchronize(device=device)

        # Give rank-0 polling thread a few ticks to observe the spread,
        # evaluate, and emit the warning.
        time.sleep(1.0)

        c10d.destroy_process_group()
        self._stderr_file.flush()
        self._stderr_file.close()

        if self.rank == 0:
            with open(stderr_path) as f:
                contents = f.read()
            self.assertIn(
                "[NCCL STRAGGLER]",
                contents,
                f"expected straggler warning, got: {contents!r}",
            )
            # The slow rank should be named.
            self.assertIn(
                "slow_ranks=[1]",
                contents,
                f"expected rank 1 to be flagged, got: {contents!r}",
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_no_straggler_no_warning(self):
        os.environ["TORCH_NCCL_STRAGGLER_DETECT"] = "1"
        os.environ["TORCH_NCCL_STRAGGLER_THRESHOLD_MS"] = "500"
        os.environ["TORCH_NCCL_STRAGGLER_POLL_MS"] = "50"

        stderr_path = self._capture_stderr_path()
        self._redirect_stderr_to_file(stderr_path)

        self._init_pg()
        device = self.device
        torch.cuda.set_device(device)

        # Both ranks enter without delay.
        x = torch.ones(4, device=device)
        for _ in range(3):
            c10d.all_reduce(x)
        torch.cuda.synchronize(device=device)
        time.sleep(0.5)

        c10d.destroy_process_group()
        self._stderr_file.flush()
        self._stderr_file.close()

        if self.rank == 0:
            with open(stderr_path) as f:
                contents = f.read()
            self.assertNotIn(
                "[NCCL STRAGGLER]",
                contents,
                f"unexpected straggler warning, got: {contents!r}",
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_disabled_no_warning(self):
        # Detector off: even a large lag must not produce a warning.
        os.environ.pop("TORCH_NCCL_STRAGGLER_DETECT", None)

        stderr_path = self._capture_stderr_path()
        self._redirect_stderr_to_file(stderr_path)

        self._init_pg()
        device = self.device
        torch.cuda.set_device(device)

        if self.rank == 1:
            time.sleep(STRAGGLER_SLEEP_S)

        x = torch.ones(4, device=device)
        c10d.all_reduce(x)
        torch.cuda.synchronize(device=device)
        time.sleep(0.5)

        c10d.destroy_process_group()
        self._stderr_file.flush()
        self._stderr_file.close()

        if self.rank == 0:
            with open(stderr_path) as f:
                contents = f.read()
            self.assertNotIn(
                "[NCCL STRAGGLER]",
                contents,
                f"detector should be off, got: {contents!r}",
            )


if __name__ == "__main__":
    run_tests()
