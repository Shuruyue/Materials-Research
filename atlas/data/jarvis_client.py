"""
JARVIS-DFT Local Data Client

Uses the JARVIS-DFT database (NIST) — a freely downloadable dataset of
~76,000 materials with DFT-computed properties.

Optimization:
- Robust download with resume capability and progress bar (tqdm)
- Automatic retry on network failure
- Local caching
"""

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from atlas.config import get_config

# Configure logging
logger = logging.getLogger(__name__)

JARVIS_DFT_3D_URL = "https://figshare.com/ndownloader/files/40357663" # Direct link to dft_3d.json


class JARVISClient:
    """
    JARVIS-DFT local data client.

    Downloads the full JARVIS-DFT dataset on first use (one-time ~500MB).
    Subsequent calls use cached local data. No API key needed.
    """

    def __init__(self):
        cfg = get_config()
        self.cache_dir = cfg.paths.raw_dir / "jarvis_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dft_3d = None

    # ------------------------------------------------------------------
    # Numerical helpers (algorithmic utilities)
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_nonnegative_finite(value: float, *, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(parsed):
            return float(default)
        return max(parsed, 0.0)

    @staticmethod
    def _coerce_positive_finite(
        value: float,
        *,
        default: float,
        floor: float = 1e-8,
    ) -> float:
        parsed = JARVISClient._coerce_nonnegative_finite(value, default=default)
        return max(parsed, float(floor))

    @staticmethod
    def _coerce_probability(value: float, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        if not np.isfinite(parsed):
            parsed = float(default)
        return float(np.clip(parsed, 0.0, 1.0))

    @staticmethod
    def _coerce_optional_finite(value: float | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return float(parsed)

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-arr))

    @staticmethod
    def _normal_cdf(x: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=12.0, neginf=-12.0)
        flat = np.ravel(arr)
        out = np.asarray(
            [0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in flat],
            dtype=float,
        )
        return out.reshape(arr.shape)

    @staticmethod
    def _robust_zscore(values: np.ndarray) -> np.ndarray:
        """Robust z-score using median/MAD scaling."""
        if values.size == 0:
            return values
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if not np.isfinite(mad) or mad < 1e-12:
            return np.zeros_like(values, dtype=float)
        sigma = 1.4826 * mad
        return (values - med) / sigma

    def _estimate_ehull_noise_series(
        self,
        df: pd.DataFrame,
        *,
        base_noise: float,
        noise_mode: str,
        adaptive_slope: float,
    ) -> np.ndarray:
        """Estimate per-row Ehull uncertainty scale.

        noise_mode:
        - constant: sigma_i = base_noise
        - adaptive: sigma_i = base_noise * (1 + slope * |z_fe|)
          where z_fe is robust z-score of formation_energy_peratom.
        """
        n = len(df)
        sigma0 = self._coerce_positive_finite(base_noise, default=0.03, floor=1e-8)
        sigma = np.full(n, sigma0, dtype=float)
        mode = str(noise_mode).strip().lower()
        if mode not in {"constant", "adaptive"}:
            raise ValueError(
                f"Unknown noise_mode: {noise_mode}. Expected constant/adaptive."
            )
        if mode == "constant" or "formation_energy_peratom" not in df.columns:
            return sigma

        fe = pd.to_numeric(df["formation_energy_peratom"], errors="coerce").to_numpy(
            dtype=float
        )
        z = np.zeros_like(fe, dtype=float)
        valid = np.isfinite(fe)
        if np.any(valid):
            z_valid = self._robust_zscore(fe[valid])
            z[valid] = np.abs(z_valid)
        slope = self._coerce_nonnegative_finite(adaptive_slope, default=0.35)
        sigma = sigma * (1.0 + slope * z)
        return np.clip(sigma, 1e-8, 1.0)

    def _stability_probability(
        self,
        df: pd.DataFrame,
        *,
        ehull_max: float,
        ehull_noise: float,
        noise_mode: str = "constant",
        adaptive_slope: float = 0.35,
    ) -> pd.Series:
        """Estimate P(Ehull <= ehull_max) from a Gaussian uncertainty model.

        Reference:
        - Sun et al., Sci. Adv. 2016 (metastability and Ehull thresholds).
        """
        if "ehull" not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        ehull = pd.to_numeric(df["ehull"], errors="coerce")
        ehull_max_v = self._coerce_nonnegative_finite(ehull_max, default=0.1)
        sigma = self._estimate_ehull_noise_series(
            df,
            base_noise=ehull_noise,
            noise_mode=noise_mode,
            adaptive_slope=adaptive_slope,
        )
        z = (ehull_max_v - ehull.to_numpy(dtype=float)) / sigma
        probs = self._normal_cdf(z)
        probs = np.clip(probs, 0.0, 1.0)
        return pd.Series(probs, index=df.index, dtype=float).where(ehull.notna(), np.nan)

    @staticmethod
    def _element_fractions_from_atoms(atoms_obj: object) -> dict[str, float]:
        """Extract element fraction vector from JARVIS atoms dict when possible."""
        if not isinstance(atoms_obj, dict):
            return {}
        elements = atoms_obj.get("elements")
        if not isinstance(elements, list) or not elements:
            return {}
        clean = [str(e).strip() for e in elements if str(e).strip()]
        if not clean:
            return {}
        counts = Counter(clean)
        total = float(sum(counts.values()))
        if total <= 0:
            return {}
        return {k: (v / total) for k, v in counts.items()}

    def _composition_feature_matrix(
        self,
        df: pd.DataFrame,
        *,
        top_k: int,
    ) -> np.ndarray:
        if "atoms" not in df.columns or len(df) == 0:
            return np.zeros((len(df), 0), dtype=float)

        all_fracs: list[dict[str, float]] = []
        elem_freq: Counter[str] = Counter()
        for atoms_obj in df["atoms"].tolist():
            fracs = self._element_fractions_from_atoms(atoms_obj)
            all_fracs.append(fracs)
            elem_freq.update(fracs.keys())

        if not elem_freq:
            return np.zeros((len(df), 0), dtype=float)

        try:
            top_k_i = int(top_k)
        except (TypeError, ValueError):
            top_k_i = 24
        ordered = [k for k, _ in elem_freq.most_common(max(1, top_k_i))]
        mat = np.zeros((len(df), len(ordered)), dtype=float)
        for i, fracs in enumerate(all_fracs):
            for j, elem in enumerate(ordered):
                mat[i, j] = float(fracs.get(elem, 0.0))
        return mat

    @staticmethod
    def _normalize_features(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        center = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - center), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        return (arr - center) / scale

    def _feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: tuple[str, ...],
        *,
        feature_space: str = "property",
        composition_top_k: int = 24,
    ) -> np.ndarray:
        canon = str(feature_space).strip().lower()
        if canon not in {"property", "composition", "hybrid"}:
            raise ValueError(
                f"Unknown feature_space: {feature_space}. Expected property/composition/hybrid."
            )

        prop_x = np.zeros((len(df), 0), dtype=float)
        if canon in {"property", "hybrid"}:
            cols = [c for c in feature_cols if c in df.columns]
            if cols:
                work = df[cols].apply(pd.to_numeric, errors="coerce").copy()
                for col in cols:
                    med = work[col].median()
                    if pd.isna(med):
                        med = 0.0
                    work[col] = work[col].fillna(float(med))
                prop_x = work.to_numpy(dtype=float)
                prop_x = self._normalize_features(prop_x)

        comp_x = np.zeros((len(df), 0), dtype=float)
        if canon in {"composition", "hybrid"}:
            comp_x = self._composition_feature_matrix(df, top_k=composition_top_k)
            comp_x = self._normalize_features(comp_x)

        if prop_x.shape[1] == 0 and comp_x.shape[1] == 0:
            return np.arange(len(df), dtype=float).reshape(-1, 1)
        if prop_x.shape[1] == 0:
            return comp_x
        if comp_x.shape[1] == 0:
            return prop_x
        return np.concatenate([prop_x, comp_x], axis=1)

    @staticmethod
    def _kcenter_indices(
        features: np.ndarray,
        k: int,
        *,
        random_state: int,
    ) -> list[int]:
        """Greedy k-center subset selection.

        Reference:
        - Gonzalez (1985), farthest-first traversal (k-center 2-approximation).
        - Sener & Savarese (2018), coreset intuition for active learning.
        """
        feat = np.asarray(features, dtype=float)
        if feat.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {feat.shape}")
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        n = int(feat.shape[0])
        if k >= n:
            return list(range(n))
        if k <= 0 or n == 0:
            return []

        rng = np.random.RandomState(random_state)
        first = int(rng.randint(0, n))
        selected = [first]
        diff = feat - feat[first]
        min_d2 = np.einsum("ij,ij->i", diff, diff)

        for _ in range(1, k):
            nxt = int(np.argmax(min_d2))
            selected.append(nxt)
            diff = feat - feat[nxt]
            d2 = np.einsum("ij,ij->i", diff, diff)
            min_d2 = np.minimum(min_d2, d2)
        return selected

    def _sample_dataframe(
        self,
        df: pd.DataFrame,
        *,
        n: int,
        strategy: str,
        random_state: int,
        feature_cols: tuple[str, ...],
        score_col: str | None = None,
        feature_space: str = "property",
        composition_top_k: int = 24,
    ) -> pd.DataFrame:
        canon = str(strategy).strip().lower()
        if canon not in {"random", "kcenter", "hybrid"}:
            raise ValueError(
                f"Unknown sampling strategy: {strategy}. Expected one of random/kcenter/hybrid."
            )
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            return df.iloc[0:0].copy()
        if len(df) <= n:
            return df.copy()

        if canon == "random":
            return df.sample(n=n, random_state=random_state).copy()

        if canon == "kcenter":
            features = self._feature_matrix(
                df,
                feature_cols,
                feature_space=feature_space,
                composition_top_k=composition_top_k,
            )
            idx = self._kcenter_indices(features, n, random_state=random_state)
            return df.iloc[idx].copy()

        if canon == "hybrid":
            n_exploit = max(1, n // 2)
            if score_col and score_col in df.columns:
                exploit = df.sort_values(score_col, ascending=False).head(n_exploit)
            else:
                exploit = df.sample(n=n_exploit, random_state=random_state)
            remaining = df.drop(index=exploit.index)
            n_explore = n - len(exploit)
            if n_explore <= 0:
                return exploit.copy()
            if len(remaining) <= n_explore:
                return pd.concat([exploit, remaining]).copy()
            features = self._feature_matrix(
                remaining,
                feature_cols,
                feature_space=feature_space,
                composition_top_k=composition_top_k,
            )
            idx = self._kcenter_indices(features, n_explore, random_state=random_state + 1)
            explore = remaining.iloc[idx]
            return pd.concat([exploit, explore]).copy()

        return df.iloc[0:0].copy()

    def load_dft_3d(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the JARVIS-DFT 3D materials database.
        """
        cache_file = self.cache_dir / "dft_3d.pkl"
        json_file = self.cache_dir / "dft_3d.json"
        dft_3d = None

        if not force_reload and cache_file.exists():
            print(f"  Loading cached JARVIS-DFT data from {cache_file}")
            try:
                self._dft_3d = pd.read_pickle(cache_file)
                return self._dft_3d
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading from source.")

        # Check if raw JSON exists, otherwise download
        if not json_file.exists() or force_reload:
            self._download_file(JARVIS_DFT_3D_URL, json_file)

        print("  Processing JARVIS-DFT data...")
        try:
            with open(json_file, encoding="utf-8") as f:
                dft_3d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load local JSON cache: {e}. Falling back to jarvis.db.figshare.")
            try:
                from jarvis.db.figshare import data as jdata
                dft_3d = jdata("dft_3d")
            except Exception as fig_e:
                raise RuntimeError("Failed to load JARVIS-DFT data from both local cache and figshare backend.") from fig_e

        # Convert to DataFrame
        df = pd.DataFrame(dft_3d)

        # Clean numeric columns (JARVIS data has mixed str/float values)
        numeric_cols = [
            "optb88vdw_bandgap", "ehull", "spillage",
            "formation_energy_peratom", "optb88vdw_total_energy",
            "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Cache locally
        df.to_pickle(cache_file)
        print(f"  Downloaded {len(df)} materials, cached at {cache_file}")

        self._dft_3d = df
        return df

    def stats(self) -> dict:
        """
        Return compact dataset statistics for CLI reporting.
        """
        df = self.load_dft_3d()
        out = {
            "n_materials_total": int(len(df)),
            "n_with_atoms": int(df["atoms"].notna().sum()) if "atoms" in df.columns else 0,
            "n_with_bandgap": int(df["optb88vdw_bandgap"].notna().sum()) if "optb88vdw_bandgap" in df.columns else 0,
            "n_with_formation_energy": int(df["formation_energy_peratom"].notna().sum()) if "formation_energy_peratom" in df.columns else 0,
            "n_with_ehull": int(df["ehull"].notna().sum()) if "ehull" in df.columns else 0,
            "n_with_spillage": int(df["spillage"].notna().sum()) if "spillage" in df.columns else 0,
        }
        return out

    def _download_file(self, url: str, dest_path: Path, max_retries: int = 3):
        """Download file with progress bar and retries."""
        print(f"  Downloading data from {url}...")
        try:
            retries = int(max_retries)
        except (TypeError, ValueError):
            retries = 3
        retries = max(1, retries)

        for attempt in range(retries):
            temp_path: Path | None = None
            try:
                # Stream download
                with requests.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 8192  # 8KB

                    with NamedTemporaryFile(
                        mode="wb",
                        dir=str(dest_path.parent),
                        prefix=f".{dest_path.name}.",
                        suffix=".part",
                        delete=False,
                    ) as tmp:
                        temp_path = Path(tmp.name)
                        downloaded = 0
                        with tqdm(
                            desc=dest_path.name,
                            total=total_size if total_size > 0 else None,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for data in response.iter_content(block_size):
                                if not data:
                                    continue
                                size = tmp.write(data)
                                downloaded += size
                                bar.update(size)

                        tmp.flush()

                    if total_size > 0 and downloaded != total_size:
                        raise RuntimeError(
                            f"Incomplete download: expected {total_size} bytes, got {downloaded}"
                        )

                    temp_path.replace(dest_path)

                return  # Success

            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))  # Backoff
                else:
                    raise RuntimeError(f"Failed to download data after {retries} attempts") from e
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to download data after {retries} attempts") from e
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def get_stable_materials(
        self,
        ehull_max: float = 0.1,
        min_band_gap: float | None = None,
        max_band_gap: float | None = None,
        *,
        mode: str = "hard",
        min_stability_prob: float = 0.5,
        ehull_noise: float = 0.03,
        ehull_noise_mode: str = "constant",
        ehull_noise_adaptive_slope: float = 0.35,
    ) -> pd.DataFrame:
        """Filter for thermodynamically stable materials.

        mode="hard":
            Classical deterministic gate: Ehull <= ehull_max.
        mode="probabilistic":
            Uses Gaussian uncertainty model and keeps samples with
            P(Ehull <= ehull_max) >= min_stability_prob.

        Reference:
        - Sun et al., Sci. Adv. 2016 (metastability landscape and Ehull scale).
        """
        df = self.load_dft_3d()
        if "ehull" not in df.columns:
            logger.warning("Column 'ehull' not found; returning empty stable set.")
            return df.iloc[0:0].copy()

        canon_mode = str(mode).strip().lower()
        if canon_mode not in {"hard", "probabilistic"}:
            raise ValueError(
                f"Unknown stability mode: {mode}. Expected hard/probabilistic."
            )
        ehull_max_v = self._coerce_nonnegative_finite(ehull_max, default=0.1)
        min_stability_prob_v = self._coerce_probability(
            min_stability_prob,
            default=0.5,
        )
        ehull_noise_v = self._coerce_positive_finite(
            ehull_noise,
            default=0.03,
            floor=1e-8,
        )
        ehull_noise_slope_v = self._coerce_nonnegative_finite(
            ehull_noise_adaptive_slope,
            default=0.35,
        )
        min_band_gap_v = self._coerce_optional_finite(min_band_gap)
        max_band_gap_v = self._coerce_optional_finite(max_band_gap)
        if (
            min_band_gap_v is not None
            and max_band_gap_v is not None
            and min_band_gap_v > max_band_gap_v
        ):
            raise ValueError(
                "min_band_gap cannot exceed max_band_gap in get_stable_materials."
            )

        result_df = df.copy()
        if canon_mode == "hard":
            mask = result_df["ehull"].notna() & (result_df["ehull"] <= ehull_max_v)
        else:
            stab_prob = self._stability_probability(
                result_df,
                ehull_max=ehull_max_v,
                ehull_noise=ehull_noise_v,
                noise_mode=ehull_noise_mode,
                adaptive_slope=ehull_noise_slope_v,
            )
            result_df["stability_probability"] = stab_prob
            mask = stab_prob.fillna(0.0) >= min_stability_prob_v

        if min_band_gap_v is not None:
            mask &= result_df["optb88vdw_bandgap"] >= min_band_gap_v
        if max_band_gap_v is not None:
            mask &= result_df["optb88vdw_bandgap"] <= max_band_gap_v

        result = result_df[mask].copy()
        if canon_mode == "probabilistic" and "stability_probability" in result.columns:
            result = result.sort_values("stability_probability", ascending=False)
            print(
                "  Found "
                f"{len(result)} stable materials "
                f"(P(Ehull<={ehull_max_v}) >= {min_stability_prob_v:.2f}, sigma={ehull_noise_v:.3f}, mode={ehull_noise_mode})"
            )
        else:
            print(f"  Found {len(result)} stable materials (ehull <= {ehull_max_v} eV/atom)")
        return result

    def get_heavy_element_materials(
        self,
        min_atomic_number: int = 50,
        ehull_max: float = 0.1,
        *,
        stability_mode: str = "hard",
        min_stability_prob: float = 0.5,
        ehull_noise: float = 0.03,
        ehull_noise_mode: str = "constant",
        ehull_noise_adaptive_slope: float = 0.35,
    ) -> pd.DataFrame:
        """Get materials containing heavy elements (strong SOC candidates)."""
        from jarvis.core.atoms import Atoms as JAtoms
        from pymatgen.core import Element

        df = self.get_stable_materials(
            ehull_max=ehull_max,
            mode=stability_mode,
            min_stability_prob=min_stability_prob,
            ehull_noise=ehull_noise,
            ehull_noise_mode=ehull_noise_mode,
            ehull_noise_adaptive_slope=ehull_noise_adaptive_slope,
        )

        def has_heavy(row):
            try:
                atoms = JAtoms.from_dict(row["atoms"])
                elements = atoms.elements
                return any(min_atomic_number <= Element(e).Z for e in elements)
            except Exception:
                return False

        print(f"  Filtering for heavy elements (Z >= {min_atomic_number})...")
        mask = df.apply(has_heavy, axis=1)
        result = df[mask].copy()
        print(f"  Found {len(result)} materials with heavy elements")
        return result

    def get_topological_materials(
        self,
        *,
        mode: str = "hard",
        spillage_threshold: float = 0.5,
        min_topology_prob: float = 0.55,
        spillage_temperature: float = 0.10,
        ehull_max: float = 0.1,
        ehull_noise: float = 0.03,
        ehull_noise_mode: str = "constant",
        ehull_noise_adaptive_slope: float = 0.35,
        low_gap_center: float = 0.3,
        low_gap_scale: float = 0.2,
        fusion_weights: tuple[float, float, float] = (0.70, 0.20, 0.10),
        score_calibration: str = "none",
        calibration_temperature: float = 1.0,
        top_quantile: float | None = None,
    ) -> pd.DataFrame:
        """Get materials classified as topologically nontrivial.

        mode="hard":
            spillage > threshold.
        mode="probabilistic":
            Geometric fusion of three probabilities:
            - p_spillage from logistic thresholding
            - p_stability from Ehull uncertainty model
            - p_low_gap from low-gap prior

            Optional calibration:
            - score_calibration="temperature": applies temperature scaling
              in logit space.
            - top_quantile: if set, uses quantile threshold instead of fixed
              probability threshold.

        References:
        - Khan et al., Sci Rep 2019 (spillage as topological screening signal).
        - Sun et al., Sci Adv 2016 (metastability signal via Ehull).
        - Guo et al., ICML 2017 (temperature scaling for calibration).
        """
        df = self.load_dft_3d()
        canon_mode = str(mode).strip().lower()
        if canon_mode not in {"hard", "probabilistic"}:
            raise ValueError(
                f"Unknown topology mode: {mode}. Expected hard/probabilistic."
            )
        spillage_threshold_v = self._coerce_nonnegative_finite(
            spillage_threshold,
            default=0.5,
        )
        min_topology_prob_v = self._coerce_probability(
            min_topology_prob,
            default=0.55,
        )
        spillage_temp_v = self._coerce_positive_finite(
            spillage_temperature,
            default=0.10,
            floor=1e-8,
        )
        ehull_max_v = self._coerce_nonnegative_finite(ehull_max, default=0.1)
        ehull_noise_v = self._coerce_positive_finite(
            ehull_noise,
            default=0.03,
            floor=1e-8,
        )
        ehull_noise_slope_v = self._coerce_nonnegative_finite(
            ehull_noise_adaptive_slope,
            default=0.35,
        )
        try:
            low_gap_center_v = float(low_gap_center)
        except (TypeError, ValueError):
            low_gap_center_v = 0.3
        if not np.isfinite(low_gap_center_v):
            low_gap_center_v = 0.3
        low_gap_scale_v = self._coerce_positive_finite(
            low_gap_scale,
            default=0.2,
            floor=1e-8,
        )

        if "spillage" in df.columns:
            if canon_mode == "hard":
                topo_mask = df["spillage"].notna() & (df["spillage"] > spillage_threshold_v)
                result = df[topo_mask].copy()
                print(
                    f"  Found {len(result)} materials with spillage > {spillage_threshold_v}"
                )
                return result

            work = df.copy()
            spillage = pd.to_numeric(work["spillage"], errors="coerce")
            spillage_prob = self._sigmoid(
                (spillage.to_numpy(dtype=float) - spillage_threshold_v) / spillage_temp_v
            )
            spillage_prob = pd.Series(spillage_prob, index=work.index).where(spillage.notna(), 0.0)

            if "ehull" in work.columns:
                stability_prob = self._stability_probability(
                    work,
                    ehull_max=ehull_max_v,
                    ehull_noise=ehull_noise_v,
                    noise_mode=ehull_noise_mode,
                    adaptive_slope=ehull_noise_slope_v,
                ).fillna(0.5)
            else:
                stability_prob = pd.Series(0.5, index=work.index, dtype=float)

            if "optb88vdw_bandgap" in work.columns:
                band_gap = pd.to_numeric(work["optb88vdw_bandgap"], errors="coerce")
                low_gap_prob = self._sigmoid(
                    (low_gap_center_v - band_gap.to_numpy(dtype=float)) / low_gap_scale_v
                )
                low_gap_prob = pd.Series(low_gap_prob, index=work.index).where(
                    band_gap.notna(), 0.5
                )
            else:
                low_gap_prob = pd.Series(0.5, index=work.index, dtype=float)

            # Weighted geometric fusion (naive conditional-independence model).
            eps = 1e-9
            ws = np.asarray(fusion_weights, dtype=float)
            if ws.shape != (3,):
                raise ValueError(
                    "fusion_weights must contain exactly three values "
                    "(spillage, stability, low_gap)."
                )
            ws = np.where(np.isfinite(ws), ws, 0.0)
            ws = np.clip(ws, 0.0, None)
            if float(ws.sum()) <= 0:
                ws = np.asarray([0.70, 0.20, 0.10], dtype=float)
            ws = ws / ws.sum()
            topo_prob = np.exp(
                ws[0] * np.log(np.clip(spillage_prob.to_numpy(dtype=float), eps, 1.0))
                + ws[1]
                * np.log(np.clip(stability_prob.to_numpy(dtype=float), eps, 1.0))
                + ws[2] * np.log(np.clip(low_gap_prob.to_numpy(dtype=float), eps, 1.0))
            )
            topo_prob = np.clip(topo_prob, 0.0, 1.0)

            calibration = str(score_calibration).strip().lower()
            if calibration not in {"none", "temperature"}:
                raise ValueError(
                    f"Unknown score_calibration: {score_calibration}. Expected none/temperature."
                )
            if calibration == "temperature":
                temp_cal = self._coerce_positive_finite(
                    calibration_temperature,
                    default=1.0,
                    floor=1e-6,
                )
                logit = np.log(np.clip(topo_prob, eps, 1.0 - eps)) - np.log(
                    np.clip(1.0 - topo_prob, eps, 1.0)
                )
                topo_prob = self._sigmoid(logit / temp_cal)

            work["spillage_probability"] = spillage_prob
            work["stability_probability"] = stability_prob
            work["low_gap_probability"] = low_gap_prob
            work["topological_probability"] = np.clip(topo_prob, 0.0, 1.0)

            if top_quantile is not None:
                q = float(top_quantile)
                if not (0.0 < q < 1.0):
                    raise ValueError("top_quantile must be in (0, 1).")
                threshold = float(
                    np.quantile(work["topological_probability"].to_numpy(dtype=float), q)
                )
            else:
                threshold = min_topology_prob_v
            topo_mask = work["topological_probability"] >= threshold
            result = work[topo_mask].copy()
            result = result.sort_values("topological_probability", ascending=False)
            print(
                "  Found "
                f"{len(result)} materials with topological_probability >= {threshold:.3f}"
            )
            return result

        print("  Warning: 'spillage' column not found, returning proxy.")
        return self.get_stable_materials(max_band_gap=0.3)

    def get_structure(self, jid: str):
        """Get a pymatgen Structure for a given JARVIS ID."""
        from jarvis.core.atoms import Atoms as JAtoms

        target_jid = str(jid).strip()
        if not target_jid:
            raise ValueError("jid must be a non-empty string")

        df = self.load_dft_3d()
        if "jid" not in df.columns:
            raise ValueError("Column 'jid' not found in JARVIS-DFT dataset")
        row = df[df["jid"].astype(str).str.strip() == target_jid]

        if len(row) == 0:
            raise ValueError(f"Material {target_jid} not found in JARVIS-DFT")

        atoms = JAtoms.from_dict(row.iloc[0]["atoms"])
        return atoms.pymatgen_converter()

    def get_training_data(
        self,
        n_topo: int = 500,
        n_trivial: int = 500,
        *,
        sampling_strategy: str = "hybrid",
        random_state: int = 42,
        topological_mode: str = "probabilistic",
        min_topology_prob: float = 0.55,
        top_quantile: float | None = None,
        trivial_stability_mode: str = "probabilistic",
        min_stability_prob: float = 0.7,
        ehull_noise: float = 0.03,
        ehull_noise_mode: str = "adaptive",
        ehull_noise_adaptive_slope: float = 0.35,
        topology_fusion_weights: tuple[float, float, float] = (0.70, 0.20, 0.10),
        topology_score_calibration: str = "none",
        topology_calibration_temperature: float = 1.0,
        feature_space: str = "hybrid",
        composition_top_k: int = 24,
        diversity_features: tuple[str, ...] = (
            "spillage",
            "ehull",
            "optb88vdw_bandgap",
            "formation_energy_peratom",
        ),
    ) -> dict:
        """Prepare a balanced training dataset for ML.

        sampling_strategy:
            - random: uniform random subset
            - kcenter: farthest-first geometric diversity
            - hybrid: top-score exploitation + k-center exploration

        feature_space:
            - property: only scalar property features
            - composition: only composition-fraction features
            - hybrid: concatenated property + composition features
        """
        topo = self.get_topological_materials(
            mode=topological_mode,
            min_topology_prob=min_topology_prob,
            top_quantile=top_quantile,
            ehull_noise=ehull_noise,
            ehull_noise_mode=ehull_noise_mode,
            ehull_noise_adaptive_slope=ehull_noise_adaptive_slope,
            fusion_weights=topology_fusion_weights,
            score_calibration=topology_score_calibration,
            calibration_temperature=topology_calibration_temperature,
        )
        trivial = self.get_stable_materials(
            min_band_gap=2.0,
            max_band_gap=6.0,
            mode=trivial_stability_mode,
            min_stability_prob=min_stability_prob,
            ehull_noise=ehull_noise,
            ehull_noise_mode=ehull_noise_mode,
            ehull_noise_adaptive_slope=ehull_noise_adaptive_slope,
        )

        # Prevent label leakage: a jid cannot appear in both classes.
        if "jid" in topo.columns and "jid" in trivial.columns:
            topo_ids = set(topo["jid"].astype(str))
            trivial = trivial[~trivial["jid"].astype(str).isin(topo_ids)].copy()

        topo = self._sample_dataframe(
            topo,
            n=n_topo,
            strategy=sampling_strategy,
            random_state=random_state,
            feature_cols=diversity_features,
            score_col="topological_probability",
            feature_space=feature_space,
            composition_top_k=composition_top_k,
        )
        trivial = self._sample_dataframe(
            trivial,
            n=n_trivial,
            strategy=sampling_strategy,
            random_state=random_state + 13,
            feature_cols=diversity_features,
            score_col="stability_probability",
            feature_space=feature_space,
            composition_top_k=composition_top_k,
        )

        print("\n  Training data prepared:")
        print(f"    Topological:  {len(topo)}")
        print(f"    Trivial:      {len(trivial)}")
        print(f"    Total:        {len(topo) + len(trivial)}")

        return {
            "topo": topo,
            "trivial": trivial,
            "total": len(topo) + len(trivial),
        }
