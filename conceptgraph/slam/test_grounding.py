"""find() must score a copied index so a concurrent merge cannot leak deleted ids."""

from __future__ import annotations

import unittest
from uuid import uuid4

import numpy as np

from conceptgraph.slam.grounding import GroundingService


def _obj(name: str, ft, n_obs: int = 4, center=None):
    center = [0.0, 0.0, 0.0] if center is None else center
    return {
        "id": uuid4(),
        "class_name": name,
        "num_detections": n_obs,
        "clip_ft": np.asarray(ft, dtype=np.float32),
        "center": center,
        "aabb": [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
    }


class TestGroundingFind(unittest.TestCase):
    def test_find_ranks_matching_feature(self):
        mug = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        chair = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        svc = GroundingService(default_min_sim=0.1, default_min_obs=1)
        with svc.map_lock:
            svc.set_objects(
                [
                    _obj("mug", mug, center=[1.0, 0.0, 0.5]),
                    _obj("chair", chair, center=[0.0, 1.0, 0.0]),
                ]
            )
        hits = svc.find("mug", query_ft=mug, k=2, min_sim=0.1, min_obs=1)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].class_name, "mug")
        self.assertGreater(hits[0].sim, 0.9)

    def test_min_obs_filters(self):
        ft = np.array([1.0, 0.0], dtype=np.float32)
        svc = GroundingService()
        with svc.map_lock:
            svc.set_objects([_obj("mug", ft, n_obs=1)])
        hits = svc.find("mug", query_ft=ft, min_obs=3, min_sim=0.0)
        self.assertEqual(hits, [])

    def test_copy_survives_delete_during_score(self):
        mug = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        svc = GroundingService()
        keep = _obj("mug", mug, n_obs=4)
        gone = _obj("ghost", mug, n_obs=4)
        with svc.map_lock:
            svc.set_objects([keep, gone])
        with svc.map_lock:
            snapshot = svc._copy_index(min_obs=1)
            svc.set_objects([keep])
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot["rows"]), 2)
        gone_id = str(gone["id"])
        self.assertTrue(any(r["id"] == gone_id for r in snapshot["rows"]))
        with svc.map_lock:
            live_ids = {str(o["id"]) for o in svc.objects}
        self.assertNotIn(gone_id, live_ids)


if __name__ == "__main__":
    unittest.main()
