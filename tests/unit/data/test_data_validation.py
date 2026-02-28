"""Unit tests for atlas.data.data_validation."""

import math

import pytest

from atlas.data.data_validation import (
    _load_records_from_input,
    TrustScore,
    ValidationReport,
    check_duplicates,
    check_leakage,
    check_outliers,
    check_provenance,
    check_schema,
    compute_drift,
    compute_trust_score,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_valid_records(self):
        records = [
            {"jid": "J1", "atoms": {"coords": []}},
            {"jid": "J2", "atoms": {"coords": []}},
        ]
        violations = check_schema(records)
        assert violations == []

    def test_missing_jid(self):
        records = [{"atoms": {"coords": []}}]
        violations = check_schema(records)
        assert len(violations) == 1
        assert "jid" in violations[0]["missing"]

    def test_missing_multiple_fields(self):
        records = [{"other": 1}]
        violations = check_schema(records, required_fields=["jid", "atoms", "formula"])
        assert len(violations) == 1
        assert set(violations[0]["missing"]) == {"jid", "atoms", "formula"}

    def test_custom_required_fields(self):
        records = [{"x": 1, "y": 2}]
        violations = check_schema(records, required_fields=["x", "y", "z"])
        assert len(violations) == 1
        assert violations[0]["missing"] == ["z"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestCheckProvenance:
    def test_all_have_provenance(self):
        records = [
            {"provenance_type": "dft_primary"},
            {"provenance_type": "experimental"},
        ]
        assert check_provenance(records) == []

    def test_missing_provenance(self):
        records = [
            {"provenance_type": "dft_primary"},
            {"other": "value"},
            {"provenance_type": ""},  # empty string is falsy
        ]
        missing = check_provenance(records)
        missing_indices = {row["index"] for row in missing}
        assert 1 in missing_indices
        assert 2 in missing_indices

    def test_empty_records(self):
        assert check_provenance([]) == []


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


class TestCheckLeakage:
    def test_no_leakage(self):
        split_ids = {
            "train": {"a", "b", "c"},
            "val": {"d", "e"},
            "test": {"f", "g"},
        }
        overlap = check_leakage(split_ids)
        assert all(v == 0 for v in overlap.values())

    def test_train_val_leakage(self):
        split_ids = {
            "train": {"a", "b", "c"},
            "val": {"c", "d"},
            "test": {"e"},
        }
        overlap = check_leakage(split_ids)
        assert overlap["train__val"] == 1
        assert overlap["train__test"] == 0
        assert overlap["val__test"] == 0

    def test_full_leakage(self):
        ids = {"x", "y", "z"}
        split_ids = {"train": ids, "val": ids, "test": ids}
        overlap = check_leakage(split_ids)
        assert overlap["train__val"] == 3
        assert overlap["train__test"] == 3
        assert overlap["val__test"] == 3

    def test_missing_split(self):
        split_ids = {"train": {"a"}}
        overlap = check_leakage(split_ids)
        assert overlap["train__val"] == 0


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestCheckDuplicates:
    def test_no_duplicates(self):
        records = [{"jid": "J1"}, {"jid": "J2"}, {"jid": "J3"}]
        dupes = check_duplicates(records)
        assert len(dupes) == 0

    def test_has_duplicates(self):
        records = [{"jid": "J1"}, {"jid": "J2"}, {"jid": "J1"}]
        dupes = check_duplicates(records)
        assert "J1" in dupes
        assert len(dupes["J1"]) == 2

    def test_custom_key(self):
        records = [{"formula": "Li2O"}, {"formula": "Li2O"}]
        dupes = check_duplicates(records, key_field="formula")
        assert "Li2O" in dupes


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestCheckOutliers:
    def test_no_outliers(self):
        values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.0]
        outliers = check_outliers(values)
        assert len(outliers) == 0

    def test_clear_outlier(self):
        values = [1.0] * 100 + [1000.0]
        outliers = check_outliers(values, sigma_threshold=5.0)
        assert len(outliers) >= 1

    def test_empty_values(self):
        assert check_outliers([]) == []

    def test_with_none_values(self):
        values = [1.0, None, 1.1, float("nan"), 0.9]
        outliers = check_outliers(values)
        assert isinstance(outliers, list)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class TestComputeDrift:
    def test_same_distribution(self):
        vals = [float(i) for i in range(100)]
        kl = compute_drift(vals, vals)
        assert kl < 0.01  # should be near zero

    def test_different_distribution(self):
        old = [float(i) for i in range(100)]
        new = [float(i + 50) for i in range(100)]
        kl = compute_drift(old, new)
        assert kl > 0  # should be positive

    def test_too_few_samples(self):
        assert compute_drift([1.0, 2.0], [3.0, 4.0]) == 0.0


# ---------------------------------------------------------------------------
# Trust scoring
# ---------------------------------------------------------------------------


class TestComputeTrustScore:
    def test_dft_primary_full_metadata(self):
        sample = {
            "jid": "JVASP-1",
            "provenance_type": "dft_primary",
            "source_key": "jarvis_dft",
            "source_version": "2024.1",
            "source_id": "JVASP-1",
        }
        ts = compute_trust_score(sample)
        assert ts.provenance_points == 30
        assert ts.completeness_points == 20.0
        assert ts.total >= 70
        assert ts.tier == "benchmark"

    def test_synthetic_minimal(self):
        sample = {"jid": "SYN-1", "provenance_type": "synthetic"}
        ts = compute_trust_score(sample)
        assert ts.provenance_points == 8
        assert ts.completeness_points < 20
        assert ts.tier in ("raw", "curated")

    def test_missing_provenance(self):
        sample = {"jid": "X-1"}
        ts = compute_trust_score(sample)
        assert ts.provenance_points == 0

    def test_with_property_stats(self):
        sample = {
            "jid": "J-1",
            "provenance_type": "dft_primary",
            "source_key": "jarvis_dft",
            "source_version": "2024.1",
            "source_id": "J-1",
            "formation_energy": 0.5,
        }
        stats = {"formation_energy": {"mean": 0.0, "std": 1.0}}
        ts = compute_trust_score(
            sample, property_stats=stats, properties=["formation_energy"]
        )
        assert ts.plausibility_points > 0


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------


class TestDeterministicSeeding:
    def test_validation_report_deterministic(self):
        records = [
            {"jid": f"J{i}", "atoms": {}, "provenance_type": "dft_primary"}
            for i in range(50)
        ]
        r1 = validate_dataset(records)
        r2 = validate_dataset(records)
        assert r1.schema_violations == r2.schema_violations
        assert r1.provenance_missing == r2.provenance_missing
        assert r1.duplicate_count == r2.duplicate_count


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_gate_pass(self):
        report = ValidationReport()
        report.apply_gates()
        assert report.gate_pass is True

    def test_gate_fail_schema(self):
        report = ValidationReport(schema_violations=3)
        report.apply_gates()
        assert report.gate_pass is False
        assert any("schema" in f for f in report.gate_failures)

    def test_gate_fail_leakage(self):
        report = ValidationReport(leakage_count=1)
        report.apply_gates()
        assert report.gate_pass is False
        assert any("leakage" in f for f in report.gate_failures)

    def test_gate_strict_duplicates(self):
        report = ValidationReport(duplicate_count=5)
        report.apply_gates(strict=False)
        assert report.gate_pass is True  # non-strict: dupes are warning
        report.apply_gates(strict=True)
        assert report.gate_pass is False  # strict: dupes are failure

    def test_to_dict(self):
        report = ValidationReport(n_samples=100, schema_violations=0)
        d = report.to_dict()
        assert d["n_samples"] == 100
        assert "gate_pass" in d

    def test_to_json(self, tmp_path):
        report = ValidationReport(n_samples=50)
        report.apply_gates()
        path = report.to_json(tmp_path / "test_report.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["n_samples"] == 50

    def test_to_markdown(self, tmp_path):
        report = ValidationReport(n_samples=50, outlier_count=3)
        report.apply_gates()
        path = report.to_markdown(tmp_path / "test_report.md")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Data Validation Report" in content


# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------


class TestManifestSchemaValidation:
    def test_report_has_required_keys(self):
        report = ValidationReport(n_samples=10)
        report.apply_gates()
        d = report.to_dict()
        required = [
            "schema_version", "timestamp", "n_samples", "schema_violations",
            "provenance_missing", "leakage_count", "gate_pass",
        ]
        for key in required:
            assert key in d, f"Missing required key: {key}"


class TestInputLoader:
    def test_load_records_from_json(self, tmp_path):
        p = tmp_path / "records.json"
        p.write_text('[{"jid":"J1","atoms":{}},{"jid":"J2","atoms":{}}]', encoding="utf-8")
        rows = _load_records_from_input(p)
        assert len(rows) == 2
        assert rows[0]["jid"] == "J1"
