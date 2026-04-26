import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATH = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"
DB_PATH = r"C:\Users\vrush\Downloads\phase3_protocol_log.db"


class BHOracle:
    """
    Virtual lab oracle.

    Uses only real dataset columns:
    ligand, additive, base, aryl_halide, yield, split
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

        missing = [col for col in required_columns if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.train_df = self.df[self.df["split"] == "train"].copy()

        if self.train_df.empty:
            raise ValueError("Train split is empty.")

    def query(self, ligand: str, additive: str, base: str, aryl_halide: str) -> dict:
        exact = self.train_df[
            (self.train_df["ligand"] == ligand)
            & (self.train_df["additive"] == additive)
            & (self.train_df["base"] == base)
            & (self.train_df["aryl_halide"] == aryl_halide)
        ]

        if len(exact) > 0:
            true_yield = float(exact["yield"].mean())
            source = "exact_train_match"
            confidence = 0.95
            matched_rows = len(exact)

            if "reaction_SMILES" in exact.columns:
                reaction_smiles = exact["reaction_SMILES"].iloc[0]
            else:
                reaction_smiles = None

        else:
            fallback = self.train_df[
                (self.train_df["base"] == base)
                & (self.train_df["aryl_halide"] == aryl_halide)
            ]

            if len(fallback) > 0:
                true_yield = float(fallback["yield"].mean())
                source = "fallback_same_base_and_aryl_halide"
                confidence = 0.65
                matched_rows = len(fallback)

                if "reaction_SMILES" in fallback.columns:
                    reaction_smiles = fallback["reaction_SMILES"].iloc[0]
                else:
                    reaction_smiles = None

            else:
                true_yield = float(self.train_df["yield"].mean())
                source = "fallback_train_average"
                confidence = 0.35
                matched_rows = len(self.train_df)
                reaction_smiles = None

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
            "reaction_SMILES": reaction_smiles,
        }

    def get_low_yield_start(self) -> dict:
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
        }


class FailureClassifierAgent:
    """
    Labels the experiment result.
    """

    def run(self, result: dict, previous_yield=None) -> str:
        y = result["yield_percent"]

        if result["confidence"] < 0.5:
            return "high_uncertainty"

        if previous_yield is not None:
            improvement = y - previous_yield

            if improvement < 5 and y < 80:
                return "no_improvement"

        if y < 30:
            return "low_yield"

        if 30 <= y < 60:
            return "moderate_yield"

        if 60 <= y < 80:
            return "near_success"

        return "success"


class SimpleAnalyzer:
    """
    No-sklearn analyzer.

    It estimates parameter importance by checking how much the average yield changes
    across each parameter.

    Example:
    If different additives cause big yield changes, additive gets high importance.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()
        self.features = ["ligand", "additive", "base", "aryl_halide"]
        self.feature_importance = self.calculate_importance()

    def calculate_importance(self) -> dict:
        importance = {}

        for feature in self.features:
            grouped = self.train_df.groupby(feature)["yield"].mean()

            if len(grouped) <= 1:
                score = 0
            else:
                score = grouped.max() - grouped.min()

            importance[feature] = round(float(score), 2)

        return dict(
            sorted(
                importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def top_parameters(self, top_k: int = 3) -> list:
        return list(self.feature_importance.items())[:top_k]

    def predict_yield(self, protocol: dict) -> float:
        """
        Predict yield using a simple average of matching groups.
        This avoids sklearn.
        """

        df = self.train_df.copy()
        predictions = []

        for feature in self.features:
            subset = df[df[feature] == protocol[feature]]

            if len(subset) > 0:
                predictions.append(float(subset["yield"].mean()))

        exact = df[
            (df["ligand"] == protocol["ligand"])
            & (df["additive"] == protocol["additive"])
            & (df["base"] == protocol["base"])
            & (df["aryl_halide"] == protocol["aryl_halide"])
        ]

        if len(exact) > 0:
            predictions.append(float(exact["yield"].mean()))

        if predictions:
            return round(float(np.mean(predictions)), 2)

        return round(float(df["yield"].mean()), 2)


class DiagnosisAgent:
    """
    Explains what happened in simple words.
    """

    def run(self, failure_type: str, top_parameters: list, ord_result: dict) -> dict:
        top_text = ", ".join([name for name, score in top_parameters])

        messages = {
            "low_yield": "The reaction has low yield. The agent should change the reaction condition choices.",
            "moderate_yield": "The reaction works a little, but it needs better conditions.",
            "near_success": "The reaction is close to the target. The agent should try one stronger condition.",
            "no_improvement": "The new attempt did not improve enough. The agent should explore a different condition.",
            "high_uncertainty": "The result has low confidence. The agent should use a better-supported condition.",
            "success": "The reaction reached the target yield. No more recovery is needed.",
        }

        diagnosis = messages.get(failure_type, "The agent selected a recovery strategy.")

        if top_text:
            diagnosis += f" Most important parameters: {top_text}."

        if ord_result["most_common_successful_base"]:
            diagnosis += (
                f" Local dataset check shows successful reactions often use base: "
                f"{ord_result['most_common_successful_base']}."
            )

        return {
            "diagnosis": diagnosis,
            "failure_type": failure_type,
        }


class ADMETFilter:
    """
    Safe ADMET placeholder.

    This does not require RDKit.
    It checks that reaction_SMILES exists and logs it.
    """

    def run(self, reaction_smiles: str) -> dict:
        if reaction_smiles:
            return {
                "admet_pass": "not_checked",
                "flags": ["reaction_SMILES found. Full ADMET check can be added with RDKit later."],
            }

        return {
            "admet_pass": "not_checked",
            "flags": ["No reaction_SMILES available."],
        }


class LocalORDCorroboration:
    """
    Local ORD-style check.

    It finds the most common base in high-yield reactions for the same aryl_halide.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()

    def run(self, aryl_halide: str) -> dict:
        subset = self.train_df[
            (self.train_df["aryl_halide"] == aryl_halide)
            & (self.train_df["yield"] >= 70)
        ].copy()

        if subset.empty:
            return {
                "most_common_successful_base": None,
                "support_count": 0,
                "note": "No high-yield local matches found.",
            }

        counts = subset["base"].value_counts()

        return {
            "most_common_successful_base": counts.index[0],
            "support_count": int(counts.iloc[0]),
            "note": "Local dataset used as ORD-style corroboration.",
        }


class RecoveryAgent:
    """
    Chooses next protocol using predicted yield from SimpleAnalyzer.
    """

    def __init__(self, oracle: BHOracle, analyzer: SimpleAnalyzer):
        self.oracle = oracle
        self.analyzer = analyzer

    def run(self, current_protocol: dict, tried_protocols: list) -> dict:
        df = self.oracle.train_df.copy()

        same_aryl = df[df["aryl_halide"] == current_protocol["aryl_halide"]].copy()

        if same_aryl.empty:
            same_aryl = df.copy()

        tried_set = set()

        for p in tried_protocols:
            tried_set.add(
                (
                    p["ligand"],
                    p["additive"],
                    p["base"],
                    p["aryl_halide"],
                )
            )

        scored = []

        for _, row in same_aryl.iterrows():
            candidate = {
                "ligand": row["ligand"],
                "additive": row["additive"],
                "base": row["base"],
                "aryl_halide": row["aryl_halide"],
            }

            key = (
                candidate["ligand"],
                candidate["additive"],
                candidate["base"],
                candidate["aryl_halide"],
            )

            if key in tried_set:
                continue

            predicted = self.analyzer.predict_yield(candidate)

            scored.append(
                {
                    "protocol": candidate,
                    "predicted_yield": predicted,
                }
            )

        if not scored:
            return {
                "next_protocol": current_protocol,
                "predicted_yield": self.analyzer.predict_yield(current_protocol),
                "reason": "No untried candidate found. Retrying current protocol.",
            }

        scored = sorted(scored, key=lambda x: x["predicted_yield"], reverse=True)
        best = scored[0]

        return {
            "next_protocol": best["protocol"],
            "predicted_yield": best["predicted_yield"],
            "reason": "Selected the candidate with the best simple predicted yield.",
        }


class ValidatorAgent:
    """
    Checks valid dataset values.
    """

    def __init__(self, oracle: BHOracle):
        self.valid_values = {
            "ligand": set(oracle.train_df["ligand"].dropna().unique()),
            "additive": set(oracle.train_df["additive"].dropna().unique()),
            "base": set(oracle.train_df["base"].dropna().unique()),
            "aryl_halide": set(oracle.train_df["aryl_halide"].dropna().unique()),
        }

    def run(self, protocol: dict) -> dict:
        errors = []

        for key in ["ligand", "additive", "base", "aryl_halide"]:
            if key not in protocol:
                errors.append(f"Missing key: {key}")
            elif protocol[key] not in self.valid_values[key]:
                errors.append(f"Invalid value for {key}: {protocol[key]}")

        return {
            "approved": len(errors) == 0,
            "errors": errors,
        }


class MemoryLogger:
    """
    Saves every attempt into SQLite.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.create_table()

    def create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt INTEGER,
                ligand TEXT,
                additive TEXT,
                base TEXT,
                aryl_halide TEXT,
                yield_percent REAL,
                true_yield_percent REAL,
                failure_type TEXT,
                diagnosis TEXT,
                admet_pass TEXT,
                admet_flags TEXT,
                top_parameters TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    def log_attempt(
        self,
        attempt: int,
        protocol: dict,
        result: dict,
        failure_type: str,
        diagnosis: dict,
        admet_result: dict,
        top_parameters: list,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO protocol_log (
                attempt,
                ligand,
                additive,
                base,
                aryl_halide,
                yield_percent,
                true_yield_percent,
                failure_type,
                diagnosis,
                admet_pass,
                admet_flags,
                top_parameters
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt,
                str(protocol["ligand"]),
                str(protocol["additive"]),
                str(protocol["base"]),
                str(protocol["aryl_halide"]),
                float(result["yield_percent"]),
                float(result["true_yield_percent"]),
                str(failure_type),
                str(diagnosis["diagnosis"]),
                str(admet_result.get("admet_pass")),
                "; ".join(admet_result.get("flags", [])),
                "; ".join([f"{name}:{score}" for name, score in top_parameters]),
            ),
        )

        conn.commit()
        conn.close()


class Phase3Loop:
    """
    Phase 3 intelligence loop.
    """

    def __init__(self, csv_path: str, db_path: str):
        self.oracle = BHOracle(csv_path)
        self.classifier = FailureClassifierAgent()
        self.analyzer = SimpleAnalyzer(self.oracle.train_df)
        self.diagnosis = DiagnosisAgent()
        self.admet = ADMETFilter()
        self.ord_check = LocalORDCorroboration(self.oracle.train_df)
        self.recovery = RecoveryAgent(self.oracle, self.analyzer)
        self.validator = ValidatorAgent(self.oracle)
        self.memory = MemoryLogger(db_path)

        self.success_threshold = 80.0
        self.max_attempts = 3

    def run(self):
        print("=" * 70)
        print("PHASE 3: INTELLIGENCE LAYER")
        print("=" * 70)

        print("\nWhat Phase 3 adds:")
        print("- Parameter importance without sklearn")
        print("- Diagnosis Agent")
        print("- ADMET-safe placeholder")
        print("- Local ORD-style corroboration")
        print("- SQLite memory logging")
        print("- Model-style recovery using simple predicted yield")

        print("\nTop Parameters")
        print("-" * 70)

        for name, score in self.analyzer.top_parameters(4):
            print(f"{name}: {score}")

        current_protocol = self.oracle.get_low_yield_start()
        previous_yield = None
        history = []

        for attempt in range(1, self.max_attempts + 1):
            print("\n" + "=" * 70)
            print(f"ATTEMPT {attempt}")
            print("=" * 70)

            print("\nProtocol")
            print("-" * 70)
            self.print_protocol(current_protocol)

            validation = self.validator.run(current_protocol)

            if not validation["approved"]:
                print("\nValidator failed:")
                print(validation["errors"])
                break

            result = self.oracle.query(
                ligand=current_protocol["ligand"],
                additive=current_protocol["additive"],
                base=current_protocol["base"],
                aryl_halide=current_protocol["aryl_halide"],
            )

            print("\nSimulator Result")
            print("-" * 70)
            print(f"yield_percent: {result['yield_percent']}")
            print(f"true_yield_percent: {result['true_yield_percent']}")
            print(f"confidence: {result['confidence']}")
            print(f"source: {result['source']}")

            failure_type = self.classifier.run(result, previous_yield)

            print("\nFailure Classifier")
            print("-" * 70)
            print(f"failure_type: {failure_type}")

            top_parameters = self.analyzer.top_parameters(3)

            print("\nAnalyzer Output")
            print("-" * 70)
            for name, score in top_parameters:
                print(f"{name}: {score}")

            ord_result = self.ord_check.run(current_protocol["aryl_halide"])

            print("\nLocal ORD-Style Corroboration")
            print("-" * 70)
            print(f"most_common_successful_base: {ord_result['most_common_successful_base']}")
            print(f"support_count: {ord_result['support_count']}")
            print(f"note: {ord_result['note']}")

            diagnosis_result = self.diagnosis.run(
                failure_type=failure_type,
                top_parameters=top_parameters,
                ord_result=ord_result,
            )

            print("\nDiagnosis Agent")
            print("-" * 70)
            print(diagnosis_result["diagnosis"])

            admet_result = self.admet.run(result.get("reaction_SMILES"))

            print("\nADMET Filter")
            print("-" * 70)
            print(f"admet_pass: {admet_result.get('admet_pass')}")
            print(f"flags: {admet_result.get('flags')}")

            self.memory.log_attempt(
                attempt=attempt,
                protocol=current_protocol,
                result=result,
                failure_type=failure_type,
                diagnosis=diagnosis_result,
                admet_result=admet_result,
                top_parameters=top_parameters,
            )

            history.append(
                {
                    "attempt": attempt,
                    "protocol": current_protocol,
                    "result": result,
                    "failure_type": failure_type,
                }
            )

            if result["yield_percent"] >= self.success_threshold:
                print("\nStop Policy")
                print("-" * 70)
                print(f"Success. Yield reached {result['yield_percent']}%.")
                break

            if attempt == self.max_attempts:
                print("\nStop Policy")
                print("-" * 70)
                print("Stopped. Max attempts reached.")
                break

            tried_protocols = [item["protocol"] for item in history]

            recovery_result = self.recovery.run(
                current_protocol=current_protocol,
                tried_protocols=tried_protocols,
            )

            print("\nRecovery Agent")
            print("-" * 70)
            print(f"predicted_yield: {recovery_result['predicted_yield']}")
            print(f"reason: {recovery_result['reason']}")

            current_protocol = recovery_result["next_protocol"]
            previous_yield = result["yield_percent"]

        print("\n" + "=" * 70)
        print("PHASE 3 COMPLETE")
        print("=" * 70)

        print(f"\nSQLite memory saved here:")
        print(DB_PATH)

        print("\nFinal History")
        print("-" * 70)

        for item in history:
            print(
                f"Attempt {item['attempt']}: "
                f"yield={item['result']['yield_percent']}%, "
                f"failure_type={item['failure_type']}"
            )

    def print_protocol(self, protocol: dict):
        print(f"ligand: {protocol['ligand']}")
        print(f"additive: {protocol['additive']}")
        print(f"base: {protocol['base']}")
        print(f"aryl_halide: {protocol['aryl_halide']}")


def main():
    loop = Phase3Loop(CSV_PATH, DB_PATH)
    loop.run()


if __name__ == "__main__":
    main()