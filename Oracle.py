import pandas as pd
import numpy as np
from pathlib import Path


class BHOracle:
    """
    Buchwald-Hartwig reaction oracle.

    This uses the real dataset columns:
    ligand, additive, base, aryl_halide, yield, split

    It does NOT use temperature, time, solvent, or catalyst loading.
    """

    def __init__(self, csv_path: str, noise_std_percent: float = 3.0, seed: int = 42):
        self.csv_path = Path(csv_path)
        self.noise_std_percent = noise_std_percent
        self.rng = np.random.default_rng(seed)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = [
            "ligand",
            "additive",
            "base",
            "aryl_halide",
            "yield",
            "split",
        ]

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.train_df = self.df[self.df["split"] == "train"].copy()
        self.valid_df = self.df[self.df["split"] == "valid"].copy()
        self.test_df = self.df[self.df["split"] == "test"].copy()

        if self.train_df.empty:
            raise ValueError("Train split is empty. Check the split column.")

    def dataset_report(self) -> dict:
        """
        Gives basic dataset information.
        """

        total_rows = len(self.df)
        train_rows = len(self.train_df)
        valid_rows = len(self.valid_df)
        test_rows = len(self.test_df)

        mean_yield = self.df["yield"].mean()
        low_yield_percent = (self.df["yield"] < 30).mean() * 100
        high_yield_percent = (self.df["yield"] >= 80).mean() * 100

        return {
            "total_rows": total_rows,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "test_rows": test_rows,
            "mean_yield": round(mean_yield, 2),
            "low_yield_under_30_percent": round(low_yield_percent, 2),
            "high_yield_80_or_more_percent": round(high_yield_percent, 2),
            "unique_ligands": self.df["ligand"].nunique(),
            "unique_additives": self.df["additive"].nunique(),
            "unique_bases": self.df["base"].nunique(),
            "unique_aryl_halides": self.df["aryl_halide"].nunique(),
        }

    def query(self, ligand: str, additive: str, base: str, aryl_halide: str) -> dict:
        """
        Query the oracle.

        Input:
            ligand
            additive
            base
            aryl_halide

        Output:
            simulated yield result
        """

        exact_match = self.train_df[
            (self.train_df["ligand"] == ligand)
            & (self.train_df["additive"] == additive)
            & (self.train_df["base"] == base)
            & (self.train_df["aryl_halide"] == aryl_halide)
        ]

        if len(exact_match) > 0:
            true_yield = float(exact_match["yield"].mean())
            source = "exact_train_match"
            confidence = 0.95
            matched_rows = len(exact_match)

        else:
            similar_match = self.train_df[
                (self.train_df["base"] == base)
                & (self.train_df["aryl_halide"] == aryl_halide)
            ]

            if len(similar_match) > 0:
                true_yield = float(similar_match["yield"].mean())
                source = "fallback_same_base_and_aryl_halide"
                confidence = 0.65
                matched_rows = len(similar_match)

            else:
                true_yield = float(self.train_df["yield"].mean())
                source = "fallback_train_average"
                confidence = 0.35
                matched_rows = len(self.train_df)

        noisy_yield = true_yield + self.rng.normal(0, self.noise_std_percent)
        noisy_yield = float(np.clip(noisy_yield, 0, 100))

        return {
            "ligand": ligand,
            "additive": additive,
            "base": base,
            "aryl_halide": aryl_halide,
            "yield_percent": round(noisy_yield, 2),
            "true_yield_percent": round(true_yield, 2),
            "confidence": confidence,
            "source": source,
            "matched_rows": matched_rows,
        }

    def get_low_yield_start(self) -> dict:
        """
        Pick one low-yield reaction to start the demo.
        This represents a failed lab attempt.
        """

        low_yield_df = self.train_df[self.train_df["yield"] < 30].copy()

        if low_yield_df.empty:
            row = self.train_df.sample(1, random_state=42).iloc[0]
        else:
            row = low_yield_df.sample(1, random_state=42).iloc[0]

        return {
            "ligand": row["ligand"],
            "additive": row["additive"],
            "base": row["base"],
            "aryl_halide": row["aryl_halide"],
            "yield_percent": round(float(row["yield"]), 2),
        }

    def get_best_known_for_same_aryl_halide(self, aryl_halide: str, top_k: int = 5):
        """
        Finds better known reaction conditions for the same aryl_halide.

        This is useful for Phase 1 because it proves that the oracle can support
        a recovery loop.
        """

        subset = self.train_df[self.train_df["aryl_halide"] == aryl_halide].copy()

        if subset.empty:
            subset = self.train_df.copy()

        subset = subset.sort_values(by="yield", ascending=False)

        results = []

        for _, row in subset.head(top_k).iterrows():
            results.append(
                {
                    "ligand": row["ligand"],
                    "additive": row["additive"],
                    "base": row["base"],
                    "aryl_halide": row["aryl_halide"],
                    "known_yield_percent": round(float(row["yield"]), 2),
                }
            )

        return results