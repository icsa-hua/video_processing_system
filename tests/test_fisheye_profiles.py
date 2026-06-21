from __future__ import annotations

import unittest

import numpy as np

from obs_system.application_module.dummy_application.pipeline_config import PipelineConfig
from obs_system.logic_module.dummy_logic.fisheye import FishEyeProjection
from obs_system.utils.global_config import (
    FISHEYE_N_VIEWS,
    FISHEYE_VIEW_FOV_DEG,
    FISHEYE_VIEW_TILT_DEG,
    FISHEYE_VIEW_YAWS_DEG,
)


class FishEyeProfileConfigurationTests(unittest.TestCase):
    def test_default_profile_preserves_existing_geometry_and_gating(self) -> None:
        projection = FishEyeProjection(profile_name="default")

        self.assertEqual(projection.n_views, FISHEYE_N_VIEWS)
        self.assertEqual(projection.view_yaws_deg, [float(v) for v in FISHEYE_VIEW_YAWS_DEG])
        self.assertEqual(
            projection.view_tilts_deg,
            [float(FISHEYE_VIEW_TILT_DEG)] * FISHEYE_N_VIEWS,
        )
        self.assertEqual(projection.view_fov_deg, float(FISHEYE_VIEW_FOV_DEG))
        self.assertEqual(projection.full_refresh_interval_frames, 0)
        self.assertEqual(projection.view_persist_frames, 0)
        self.assertEqual(projection.backproject_edge_samples, 1)
        self.assertEqual(projection.allowed_class_names, frozenset())

        for frame_id in (0, 1, 50, 100):
            self.assertFalse(projection.is_refresh_frame(frame_id))
            self.assertFalse(projection.should_force_frame(frame_id))

    def test_jetson_profile_is_six_view_and_refreshes_every_two_seconds(self) -> None:
        projection = FishEyeProjection(profile_name="jetson_2_road")

        self.assertEqual(projection.n_views, 6)
        self.assertEqual(len(projection.view_yaws_deg), 6)
        self.assertEqual(len(projection.view_tilts_deg), 6)
        self.assertEqual(projection.full_refresh_interval_frames, 50)
        self.assertEqual(projection.view_persist_frames, 8)
        self.assertEqual(projection.backproject_edge_samples, 5)
        self.assertIn("person", projection.allowed_class_names)
        self.assertIn("car", projection.allowed_class_names)
        self.assertIn("truck", projection.allowed_class_names)

        self.assertTrue(projection.is_refresh_frame(0))
        self.assertFalse(projection.is_refresh_frame(49))
        self.assertTrue(projection.is_refresh_frame(50))

    def test_detection_hit_keeps_only_the_profiled_view_alive(self) -> None:
        projection = FishEyeProjection(profile_name="jetson_2_road")

        projection.mark_view_detection(view_id=2, frame_id=100)
        self.assertTrue(projection.has_persisted_views(108))
        self.assertFalse(projection.has_persisted_views(109))
        self.assertTrue(projection.should_force_frame(105))

    def test_refresh_alone_does_not_extend_persistence(self) -> None:
        projection = FishEyeProjection(profile_name="jetson_2_road")
        frame = np.zeros((720, 720, 3), dtype=np.uint8)

        projection.get_views(
            frame,
            fg_mask_small=None,
            frame_id=50,
            force_refresh=True,
        )

        self.assertFalse(projection.has_persisted_views(51))

    def test_pipeline_rejects_unknown_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown fisheye_profile"):
            PipelineConfig(
                model_name="model.pt",
                fisheye_profile="missing",
            ).validate()


class FishEyeGeometryTests(unittest.TestCase):
    @staticmethod
    def _empty_motion_mask() -> np.ndarray:
        return np.zeros((180, 180), dtype=np.uint8)

    def test_default_profile_keeps_all_views_when_no_mask_is_available(self) -> None:
        projection = FishEyeProjection(profile_name="default")
        frame = np.zeros((720, 720, 3), dtype=np.uint8)

        views = projection.get_views(frame, fg_mask_small=None, frame_id=1)

        self.assertEqual(len(views), FISHEYE_N_VIEWS)
        self.assertTrue(all(active and image is not None for image, _, active in views))

    def test_jetson_refresh_submits_exactly_six_views(self) -> None:
        projection = FishEyeProjection(profile_name="jetson_2_road")
        frame = np.zeros((720, 720, 3), dtype=np.uint8)

        views = projection.get_views(
            frame,
            fg_mask_small=self._empty_motion_mask(),
            frame_id=50,
            force_refresh=True,
        )

        self.assertEqual(len(views), 6)
        self.assertTrue(all(active and image is not None for image, _, active in views))

    def test_backprojected_boxes_are_finite_ordered_and_bounded(self) -> None:
        projection = FishEyeProjection(profile_name="jetson_2_road")
        frame = np.zeros((720, 720, 3), dtype=np.uint8)
        projection.get_views(frame, fg_mask_small=None, frame_id=1)

        boxes = np.array(
            [
                [150.0, 180.0, 330.0, 410.0],
                [300.0, 250.0, 500.0, 520.0],
            ],
            dtype=np.float32,
        )
        projected = projection.backproject_view_boxes(boxes, view_id=2)

        self.assertEqual(projected.shape, (2, 4))
        self.assertTrue(np.isfinite(projected).all())
        self.assertTrue((projected[:, 0] >= 0.0).all())
        self.assertTrue((projected[:, 1] >= 0.0).all())
        self.assertTrue((projected[:, 2] <= 719.0).all())
        self.assertTrue((projected[:, 3] <= 719.0).all())
        self.assertTrue((projected[:, 2] > projected[:, 0]).all())
        self.assertTrue((projected[:, 3] > projected[:, 1]).all())


if __name__ == "__main__":
    unittest.main()
