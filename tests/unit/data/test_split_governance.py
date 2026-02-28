"""Unit tests for atlas.data.split_governance."""


from atlas.data.split_governance import (
    SplitManifest,
    build_assignment_records,
    compositional_split,
    compute_split_overlap_counts,
    generate_manifest,
    iid_split,
    prototype_split,
)

# ---------------------------------------------------------------------------
# IID split
# ---------------------------------------------------------------------------


class TestIIDSplit:
    def test_basic_split(self):
        ids = [f"S{i}" for i in range(100)]
        splits = iid_split(ids, seed=42)
        assert set(splits.keys()) == {"train", "val", "test"}
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 100

    def test_ratio_approximate(self):
        ids = [f"S{i}" for i in range(1000)]
        splits = iid_split(ids, seed=42, ratios=(0.8, 0.1, 0.1))
        assert abs(len(splits["train"]) - 800) < 10
        assert abs(len(splits["val"]) - 100) < 10
        assert abs(len(splits["test"]) - 100) < 10

    def test_no_overlap(self):
        ids = [f"S{i}" for i in range(100)]
        splits = iid_split(ids, seed=42)
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_deterministic(self):
        ids = [f"S{i}" for i in range(100)]
        s1 = iid_split(ids, seed=42)
        s2 = iid_split(ids, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_different_seeds_differ(self):
        ids = [f"S{i}" for i in range(100)]
        s1 = iid_split(ids, seed=42)
        s2 = iid_split(ids, seed=99)
        assert s1["train"] != s2["train"]


# ---------------------------------------------------------------------------
# Compositional split
# ---------------------------------------------------------------------------


class TestCompositionalSplit:
    def test_basic_split(self):
        ids = [f"S{i}" for i in range(6)]
        formulas = ["Li2O", "Li2O", "Fe2O3", "Fe2O3", "NaCl", "NaCl"]
        splits = compositional_split(ids, formulas, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 6

    def test_no_overlap(self):
        ids = [f"S{i}" for i in range(100)]
        formulas = [f"Element{i % 10}O" for i in range(100)]
        splits = compositional_split(ids, formulas, seed=42)
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_deterministic(self):
        ids = [f"S{i}" for i in range(50)]
        formulas = [f"El{i % 8}O" for i in range(50)]
        s1 = compositional_split(ids, formulas, seed=42)
        s2 = compositional_split(ids, formulas, seed=42)
        assert sorted(s1["train"]) == sorted(s2["train"])
        assert sorted(s1["test"]) == sorted(s2["test"])

    def test_groups_stay_together(self):
        """All samples with the same formula should be in the same split."""
        ids = [f"S{i}" for i in range(20)]
        formulas = ["Li2O"] * 5 + ["Fe2O3"] * 5 + ["NaCl"] * 5 + ["CuO"] * 5
        splits = compositional_split(ids, formulas, seed=42)
        # Li2O samples (indices 0-4) should all be in the same split
        li_ids = set(ids[:5])
        in_train = li_ids & set(splits["train"])
        in_val = li_ids & set(splits["val"])
        in_test = li_ids & set(splits["test"])
        # All should be in exactly one split
        assert sum(len(s) > 0 for s in [in_train, in_val, in_test]) == 1


# ---------------------------------------------------------------------------
# Prototype split
# ---------------------------------------------------------------------------


class TestPrototypeSplit:
    def test_basic_split(self):
        ids = [f"S{i}" for i in range(100)]
        sgs = [str(i % 10) for i in range(100)]
        splits = prototype_split(ids, sgs, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 100

    def test_no_overlap(self):
        ids = [f"S{i}" for i in range(100)]
        sgs = [str(i % 15) for i in range(100)]
        splits = prototype_split(ids, sgs, seed=42)
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert len(train & val) == 0
        assert len(train & test) == 0

    def test_deterministic(self):
        ids = [f"S{i}" for i in range(50)]
        sgs = [str(i % 5) for i in range(50)]
        s1 = prototype_split(ids, sgs, seed=42)
        s2 = prototype_split(ids, sgs, seed=42)
        assert sorted(s1["train"]) == sorted(s2["train"])

    def test_groups_stay_together(self):
        """All samples with the same spacegroup should be in the same split."""
        ids = [f"S{i}" for i in range(30)]
        sgs = ["225"] * 10 + ["186"] * 10 + ["62"] * 10
        splits = prototype_split(ids, sgs, seed=42)
        sg225_ids = set(ids[:10])
        in_train = sg225_ids & set(splits["train"])
        in_val = sg225_ids & set(splits["val"])
        in_test = sg225_ids & set(splits["test"])
        assert sum(len(s) > 0 for s in [in_train, in_val, in_test]) == 1


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestSplitManifest:
    def test_manifest_generation(self):
        ids = [f"S{i}" for i in range(100)]
        splits = iid_split(ids, seed=42)
        manifest = generate_manifest("iid", splits, seed=42)
        assert manifest.schema_version == "2.0"
        assert manifest.strategy == "iid"
        assert manifest.seed == 42
        assert "train" in manifest.splits
        assert manifest.splits["train"]["n_samples"] == len(splits["train"])
        assert manifest.split_hash.startswith("sha256:")
        assert manifest.assignment_hash.startswith("sha256:")
        assert manifest.split_id

    def test_manifest_hash_deterministic(self):
        ids = [f"S{i}" for i in range(100)]
        s1 = iid_split(ids, seed=42)
        s2 = iid_split(ids, seed=42)
        m1 = generate_manifest("iid", s1, seed=42)
        m2 = generate_manifest("iid", s2, seed=42)
        assert m1.split_hash == m2.split_hash

    def test_manifest_hash_changes_with_data(self):
        ids1 = [f"S{i}" for i in range(100)]
        ids2 = [f"S{i}" for i in range(101)]
        s1 = iid_split(ids1, seed=42)
        s2 = iid_split(ids2, seed=42)
        m1 = generate_manifest("iid", s1, seed=42)
        m2 = generate_manifest("iid", s2, seed=42)
        assert m1.split_hash != m2.split_hash

    def test_to_json(self, tmp_path):
        ids = [f"S{i}" for i in range(50)]
        splits = iid_split(ids, seed=42)
        manifest = generate_manifest("iid", splits, seed=42)
        path = manifest.to_json(tmp_path / "test_manifest.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["split_strategy"] == "iid"
        assert data["schema_version"] == "2.0"
        assert "split_hash" in data

    def test_to_dict(self):
        manifest = SplitManifest(strategy="compositional", seed=99)
        d = manifest.to_dict()
        assert d["split_strategy"] == "compositional"
        assert d["seed"] == 99

    def test_assignment_rows_deterministic(self):
        splits = {
            "train": ["S2", "S1"],
            "val": ["S3"],
            "test": ["S5", "S4"],
        }
        group_by_id = {"S1": "g1", "S2": "g1", "S3": "g2", "S4": "g3", "S5": "g3"}
        rows1 = build_assignment_records(splits, group_by_id=group_by_id)
        rows2 = build_assignment_records(splits, group_by_id=group_by_id)
        assert rows1 == rows2
        assert rows1[0]["sample_id"] == "S1"

    def test_overlap_counts(self):
        splits = {
            "train": ["S1", "S2"],
            "val": ["S2", "S3"],
            "test": ["S4"],
        }
        overlap = compute_split_overlap_counts(splits)
        assert overlap["train__val"] == 1
        assert overlap["train__test"] == 0
        assert overlap["val__test"] == 0
