"""Unit tests for atlas.data.property_estimator."""

import numpy as np
import pandas as pd

from atlas.data.property_estimator import PropertyEstimator


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "jid": "J1",
                "formula": "SiO2",
                "hse_gap": 1.6,
                "mbj_bandgap": 1.4,
                "optb88vdw_bandgap": 1.2,
                "bulk_modulus_kv": 150.0,
                "shear_modulus_gv": 80.0,
                "density": 2.65,
                "poisson": 0.20,
            },
            {
                "jid": "J2",
                "formula": "Bi2Te3",
                "hse_gap": np.nan,
                "mbj_bandgap": 0.08,
                "optb88vdw_bandgap": 0.02,
                "bulk_modulus_kv": 40.0,
                "shear_modulus_gv": 18.0,
                "density": 7.7,
                "poisson": 0.30,
            },
            {
                "jid": "J3",
                "formula": "C",
                "hse_gap": np.nan,
                "mbj_bandgap": np.nan,
                "optb88vdw_bandgap": np.nan,
                "bulk_modulus_kv": 20.0,
                "shear_modulus_gv": 10.0,
                "density": 2.2,
                "poisson": 0.25,
            },
        ]
    )


class TestPropertyEstimator:
    def test_multifidelity_bandgap_fusion_and_probs(self):
        est = PropertyEstimator()
        out = est.extract_all_properties(_make_df())

        assert "bandgap_fused" in out.columns
        assert "bandgap_fused_std" in out.columns
        assert "bandgap_method_count" in out.columns
        assert "p_metal" in out.columns
        assert "p_insulator" in out.columns
        assert "conductivity_entropy" in out.columns

        row1 = out.loc[out["jid"] == "J1"].iloc[0]
        assert 1.2 <= float(row1["bandgap_fused"]) <= 1.6
        assert float(row1["bandgap_method_count"]) == 3.0
        assert float(row1["bandgap_fused_std"]) > 0

        row2 = out.loc[out["jid"] == "J2"].iloc[0]
        probs2 = [
            float(row2["p_metal"]),
            float(row2["p_semimetal"]),
            float(row2["p_semiconductor"]),
            float(row2["p_insulator"]),
        ]
        assert abs(sum(probs2) - 1.0) < 1e-6
        assert row2["conductivity_class"] in {
            "metal",
            "semimetal",
            "semiconductor",
            "insulator",
        }

        row3 = out.loc[out["jid"] == "J3"].iloc[0]
        assert row3["conductivity_class"] == "unknown"
        assert np.isnan(float(row3["p_metal"]))
        assert np.isnan(float(row3["conductivity_confidence"]))

    def test_formula_aware_atomic_mass_and_debye(self):
        est = PropertyEstimator()
        out = est.extract_all_properties(_make_df())

        assert "avg_atomic_mass_amu_est" in out.columns
        assert "debye_temperature" in out.columns
        assert "slack_kappa_index" in out.columns

        si_o = out.loc[out["jid"] == "J1"].iloc[0]
        carbon = out.loc[out["jid"] == "J3"].iloc[0]
        assert float(si_o["avg_atomic_mass_amu_est"]) > float(
            carbon["avg_atomic_mass_amu_est"]
        )
        assert np.isfinite(float(si_o["debye_temperature"]))
        assert np.isfinite(float(si_o["slack_kappa_index"]))

    def test_melting_estimate_has_uncertainty(self):
        est = PropertyEstimator()
        out = est.extract_all_properties(_make_df())
        assert "melting_point_est" in out.columns
        assert "melting_point_est_std" in out.columns
        row2 = out.loc[out["jid"] == "J2"].iloc[0]
        assert float(row2["melting_point_est"]) > 0
        assert float(row2["melting_point_est_std"]) >= 0

    def test_search_and_summary_still_work(self):
        est = PropertyEstimator()
        out = est.extract_all_properties(_make_df())
        filt = est.search(
            out,
            criteria={"conductivity_class": {"metal", "semimetal", "semiconductor"}},
            sort_by="bandgap_fused",
            ascending=True,
            max_results=10,
        )
        assert len(filt) >= 1

        summary = est.property_summary(out)
        assert "bandgap_fused" in summary

    def test_extract_works_without_formula_column(self):
        est = PropertyEstimator()
        df = _make_df().drop(columns=["formula"])
        out = est.extract_all_properties(df)
        assert len(out) == len(df)
        assert "avg_atomic_mass_amu_est" in out.columns
        assert np.isfinite(out["avg_atomic_mass_amu_est"]).all()

    def test_correlation_inflates_fusion_uncertainty(self):
        est = PropertyEstimator()
        out_uncorr = est.extract_all_properties(
            _make_df(),
            bandgap_fusion_correlation=0.0,
        )
        out_corr = est.extract_all_properties(
            _make_df(),
            bandgap_fusion_correlation=0.7,
        )
        row = out_uncorr.loc[out_uncorr["jid"] == "J1"].iloc[0]
        row_corr = out_corr.loc[out_corr["jid"] == "J1"].iloc[0]
        assert float(row_corr["bandgap_fused_std"]) >= float(row["bandgap_fused_std"])

    def test_adaptive_sigma_depends_on_formula_complexity(self):
        est = PropertyEstimator()
        df = pd.DataFrame(
            [
                {
                    "jid": "SIMPLE",
                    "formula": "C",
                    "hse_gap": 1.2,
                    "mbj_bandgap": 1.1,
                    "optb88vdw_bandgap": 1.0,
                    "bulk_modulus_kv": 100.0,
                    "shear_modulus_gv": 50.0,
                    "density": 2.0,
                },
                {
                    "jid": "COMPLEX",
                    "formula": "Ba2Bi1.5Te3.5",
                    "hse_gap": 1.2,
                    "mbj_bandgap": 1.1,
                    "optb88vdw_bandgap": 1.0,
                    "bulk_modulus_kv": 100.0,
                    "shear_modulus_gv": 50.0,
                    "density": 2.0,
                },
            ]
        )
        out_const = est.extract_all_properties(
            df,
            bandgap_sigma_mode="constant",
        )
        out_adapt = est.extract_all_properties(
            df,
            bandgap_sigma_mode="adaptive",
            bandgap_sigma_adaptive_slope=0.8,
        )
        simple_const = float(
            out_const.loc[out_const["jid"] == "SIMPLE", "bandgap_fused_std"].iloc[0]
        )
        simple_adapt = float(
            out_adapt.loc[out_adapt["jid"] == "SIMPLE", "bandgap_fused_std"].iloc[0]
        )
        complex_adapt = float(
            out_adapt.loc[out_adapt["jid"] == "COMPLEX", "bandgap_fused_std"].iloc[0]
        )
        assert abs(simple_adapt - simple_const) < 1e-8
        assert complex_adapt > simple_adapt
