"""Unit tests for atlas.data.data_validation."""



from atlas.data.data_validation import (
    ValidationReport,
    _load_records_from_input,
    _select_records_for_property_stats,
    benjamini_hochberg_qvalues,
    collect_property_values,
    check_duplicates,
    check_leakage,
    check_outliers,
    check_provenance,
    check_schema,
    compute_drift,
    compute_distribution_drift,
    compute_joint_distribution_drift,
    compute_ks_2sample,
    compute_mmd_rbf,
    compute_trust_score,
    compute_wasserstein_1d,
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


class TestDistributionMetrics:
    def test_wasserstein_identical(self):
        vals = [float(i) for i in range(50)]
        w1 = compute_wasserstein_1d(vals, vals)
        assert w1 == 0.0

    def test_wasserstein_shifted(self):
        old = [float(i) for i in range(50)]
        new = [float(i + 10) for i in range(50)]
        w1 = compute_wasserstein_1d(old, new)
        assert w1 > 0

    def test_mmd_shifted(self):
        old = [float(i) for i in range(80)]
        new = [float(i + 20) for i in range(80)]
        mmd2 = compute_mmd_rbf(old, new, max_points=128)
        assert mmd2 > 0

    def test_bh_qvalues(self):
        qvals = benjamini_hochberg_qvalues({"a": 0.001, "b": 0.02, "c": 0.9})
        assert set(qvals) == {"a", "b", "c"}
        assert qvals["a"] <= qvals["b"] <= qvals["c"]

    def test_distribution_drift_alert(self):
        baseline = {"formation_energy": [float(i) for i in range(100)]}
        current = {"formation_energy": [float(i + 50) for i in range(100)]}
        details, alerts = compute_distribution_drift(
            baseline,
            current,
            alpha=0.05,
            effect_threshold=0.05,
            max_points=256,
        )
        assert "formation_energy" in details
        assert "formation_energy" in alerts

    def test_ks_permutation_handles_ties(self):
        old = [0.0] * 40 + [1.0] * 40
        new = [0.0] * 10 + [1.0] * 70
        stat, p = compute_ks_2sample(
            old,
            new,
            method="permutation",
            n_permutations=128,
        )
        assert stat > 0
        assert p < 0.05

    def test_collect_property_values_source_fallback(self):
        records = [{"formation_energy_peratom": -1.23}]
        out = collect_property_values(records, ["formation_energy"])
        assert out["formation_energy"] == [-1.23]

    def test_joint_distribution_drift_alert(self):
        baseline = [
            {
                "jid": f"B{i}",
                "formation_energy": float(i),
                "band_gap": float(i) * 0.1,
            }
            for i in range(120)
        ]
        current = [
            {
                "jid": f"C{i}",
                "formation_energy": float(i + 40),
                "band_gap": float(i + 40) * 0.1,
            }
            for i in range(120)
        ]
        detail = compute_joint_distribution_drift(
            baseline,
            current,
            properties=["formation_energy", "band_gap"],
            alpha=0.05,
            effect_threshold=0.1,
            n_permutations=96,
            max_points=96,
        )
        assert detail["skipped"] is False
        assert detail["alert"] is True
        assert detail["energy_distance"] > 0

    def test_joint_distribution_drift_skips_when_insufficient_complete_cases(self):
        baseline = [{"jid": "B1", "formation_energy": 1.0, "band_gap": None}]
        current = [{"jid": "C1", "formation_energy": 2.0, "band_gap": 0.2}]
        detail = compute_joint_distribution_drift(
            baseline,
            current,
            properties=["formation_energy", "band_gap"],
            n_permutations=64,
        )
        assert detail["skipped"] is True
        assert detail["alert"] is False

    def test_joint_distribution_drift_handles_missing_without_complete_case_drop(self):
        baseline = []
        current = []
        for i in range(120):
            baseline.append(
                {
                    "jid": f"B{i}",
                    "formation_energy": float(i),
                    "band_gap": (float(i) * 0.1) if (i % 2 == 0) else None,
                }
            )
            current.append(
                {
                    "jid": f"C{i}",
                    "formation_energy": float(i + 30),
                    "band_gap": (float(i + 30) * 0.1) if (i % 2 == 0) else None,
                }
            )
        detail = compute_joint_distribution_drift(
            baseline,
            current,
            properties=["formation_energy", "band_gap"],
            alpha=0.05,
            effect_threshold=0.1,
            n_permutations=96,
            max_points=96,
            min_samples=24,
        )
        assert detail["skipped"] is False
        assert detail["n_old_complete_case"] < detail["n_old_raw"]
        assert detail["n_new_complete_case"] < detail["n_new_raw"]
        assert detail["energy_distance"] > 0.0


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

    def test_validation_strict_fails_on_drift_alert(self):
        records = [
            {
                "jid": f"J{i}",
                "atoms": {},
                "provenance_type": "dft_primary",
                "formation_energy": float(i + 30),
            }
            for i in range(60)
        ]
        baseline_values = {"formation_energy": [float(i) for i in range(60)]}
        report = validate_dataset(
            records,
            properties=["formation_energy"],
            baseline_property_values=baseline_values,
            strict=True,
            drift_effect_threshold=0.05,
        )
        assert report.drift_alert_count >= 1
        assert report.gate_pass is False

    def test_select_records_for_property_stats_prefers_train(self):
        records = [
            {"jid": "A", "formation_energy": 1.0},
            {"jid": "B", "formation_energy": 2.0},
            {"jid": "C", "formation_energy": 3.0},
        ]
        split_ids = {"train": {"A", "C"}, "val": {"B"}, "test": set()}
        selected, source = _select_records_for_property_stats(records, split_ids)
        assert source == "train_split"
        assert [r["jid"] for r in selected] == ["A", "C"]

    def test_validation_joint_drift_hard_gate(self):
        baseline_records = [
            {
                "jid": f"B{i}",
                "atoms": {},
                "provenance_type": "dft_primary",
                "formation_energy": float(i),
                "band_gap": float(i) * 0.1,
            }
            for i in range(120)
        ]
        current_records = [
            {
                "jid": f"C{i}",
                "atoms": {},
                "provenance_type": "dft_primary",
                "formation_energy": float(i + 30),
                "band_gap": float(i + 30) * 0.1,
            }
            for i in range(120)
        ]
        report = validate_dataset(
            current_records,
            properties=["formation_energy", "band_gap"],
            baseline_records=baseline_records,
            joint_drift_effect_threshold=0.1,
            joint_drift_permutations=96,
            joint_drift_max_points=96,
            joint_drift_hard_gate=True,
        )
        assert report.joint_drift_alert_count >= 1
        assert report.gate_pass is False


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

    def test_gate_drift_hard_gate(self):
        report = ValidationReport(drift_alert_count=2)
        report.apply_gates(drift_hard_gate=False)
        assert report.gate_pass is True
        report.apply_gates(drift_hard_gate=True)
        assert report.gate_pass is False

    def test_gate_joint_drift_hard_gate(self):
        report = ValidationReport(joint_drift_alert_count=1)
        report.apply_gates(joint_drift_hard_gate=False)
        assert report.gate_pass is True
        report.apply_gates(joint_drift_hard_gate=True)
        assert report.gate_pass is False

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
