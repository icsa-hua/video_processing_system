from __future__ import annotations

import collections
import unittest
from unittest.mock import patch

from obs_system.detection_module.interface.streaming_compressed import OptimizedStreamer


class CompletedFrameFpsTests(unittest.TestCase):
    def test_completed_batches_count_all_frames_without_per_result_bursts(self) -> None:
        samples = collections.deque([(0.0, 0)], maxlen=100)

        with patch(
            "obs_system.detection_module.interface.streaming_compressed.time.perf_counter",
            side_effect=[1.0, 2.0],
        ):
            total, sliding, average, _ = OptimizedStreamer._record_completed_frame_fps(
                samples,
                total_frames=0,
                completed_frames=16,
                stream_start=0.0,
            )
            self.assertEqual(total, 16)
            self.assertAlmostEqual(sliding, 16.0)
            self.assertAlmostEqual(average, 16.0)

            total, sliding, average, _ = OptimizedStreamer._record_completed_frame_fps(
                samples,
                total_frames=total,
                completed_frames=8,
                stream_start=0.0,
            )

        self.assertEqual(total, 24)
        self.assertAlmostEqual(sliding, 12.0)
        self.assertAlmostEqual(average, 12.0)
        self.assertEqual(len(samples), 3)

    def test_negative_completed_count_cannot_reduce_total(self) -> None:
        samples = collections.deque([(0.0, 10)], maxlen=100)

        with patch(
            "obs_system.detection_module.interface.streaming_compressed.time.perf_counter",
            return_value=1.0,
        ):
            total, _, _, _ = OptimizedStreamer._record_completed_frame_fps(
                samples,
                total_frames=10,
                completed_frames=-3,
                stream_start=0.0,
            )

        self.assertEqual(total, 10)


if __name__ == "__main__":
    unittest.main()
