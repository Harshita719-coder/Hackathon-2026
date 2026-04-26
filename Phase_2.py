import pandas as pd
import numpy as np
from pathlib import Path


CSV_PATH = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"


class BHOracle:
    """
    Phase 1 Oracle reused in Phase 2.

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

        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.train_df = self.df[self.df["split"] == "train"].copy()
        self.valid_df = self.df[self.df["split"] == "valid"].copy()
        self.test_df = self.df[self.df["split"] == "test"].copy()

        if self.train_df.empty:
            raise ValueError("Train split is empty. Check your split column.")

    def query(self, ligand: str, additive: str, base: str, aryl_halide: str) -> dict:
        """
        Simulates a lab experiment by returning yield for a proposed reaction.
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
        Selects a low-yield reaction to represent a failed lab attempt.
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
        }


class PlannerAgent:
    """
    Creates the goal and stop rules.
    """

    def run(self) -> dict:
        return {
            "objective": "maximize_yield",
            "success_threshold": 80.0,
            "max_attempts": 3,
            "allowed_parameters": ["ligand", "additive", "base", "aryl_halide"],
        }


class ProtocolGeneratorAgent:
    """
    Generates the first reaction protocol.

    For Phase 2, it starts with a known low-yield reaction.
    """

    def __init__(self, oracle: BHOracle):
        self.oracle = oracle

    def run(self) -> dict:
        starting_protocol = self.oracle.get_low_yield_start()

        return {
            "protocol": starting_protocol,
            "reason": "Start with a low-yield reaction to simulate a failed lab attempt.",
        }


class SimulatorAgent:
    """
    Runs the protocol against the oracle.
    """

    def __init__(self, oracle: BHOracle):
        self.oracle = oracle

    def run(self, protocol: dict) -> dict:
        result = self.oracle.query(
            ligand=protocol["ligand"],
            additive=protocol["additive"],
            base=protocol["base"],
            aryl_halide=protocol["aryl_halide"],
        )

        return result


class FailureClassifierAgent:
    """
    Deterministic failure classifier.

    No LLM.
    No hallucination.
    """

    def run(self, simulation_result: dict, previous_yield=None) -> str:
        y = simulation_result["yield_percent"]
        confidence = simulation_result["confidence"]

        if confidence < 0.5:
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


class RecoveryAgent:
    """
    Suggests the next reaction condition.

    Uses only real dataset columns:
    ligand, additive, base, aryl_halide

    Recovery logic:
    - Keep same aryl_halide
    - Search better ligand/additive/base combinations
    - Avoid protocols already tried
    """

    def __init__(self, oracle: BHOracle):
        self.oracle = oracle

    def run(self, current_protocol: dict, failure_type: str, tried_protocols: list) -> dict:
        train_df = self.oracle.train_df.copy()

        same_aryl = train_df[
            train_df["aryl_halide"] == current_protocol["aryl_halide"]
        ].copy()

        if same_aryl.empty:
            same_aryl = train_df.copy()

        same_aryl = same_aryl.sort_values(by="yield", ascending=False)

        tried_set = set()

        for protocol in tried_protocols:
            key = (
                protocol["ligand"],
                protocol["additive"],
                protocol["base"],
                protocol["aryl_halide"],
            )
            tried_set.add(key)

        for _, row in same_aryl.iterrows():
            candidate = {
                "ligand": row["ligand"],
                "additive": row["additive"],
                "base": row["base"],
                "aryl_halide": row["aryl_halide"],
            }

            candidate_key = (
                candidate["ligand"],
                candidate["additive"],
                candidate["base"],
                candidate["aryl_halide"],
            )

            if candidate_key not in tried_set:
                return {
                    "protocol_patch": candidate,
                    "reason": self.get_reason(failure_type, current_protocol, candidate),
                    "expected_known_yield": round(float(row["yield"]), 2),
                }

        return {
            "protocol_patch": current_protocol,
            "reason": "No untried better condition found. Retrying current best protocol.",
            "expected_known_yield": None,
        }

    def get_reason(self, failure_type: str, old_protocol: dict, new_protocol: dict) -> str:
        changes = []

        for key in ["ligand", "additive", "base", "aryl_halide"]:
            if old_protocol[key] != new_protocol[key]:
                changes.append(f"{key}: changed")

        if not changes:
            change_text = "No parameter changed."
        else:
            change_text = ", ".join(changes)

        reason_map = {
            "low_yield": "Low yield detected. The agent searches for a stronger ligand/additive/base combination for the same aryl halide.",
            "moderate_yield": "Moderate yield detected. The agent improves the reaction by changing available categorical reaction conditions.",
            "near_success": "Near-success detected. The agent tries a higher-yield known condition for the same aryl halide.",
            "no_improvement": "No strong improvement detected. The agent explores a different untried condition.",
            "high_uncertainty": "Low confidence detected. The agent chooses a better-supported condition from the training data.",
            "success": "Success threshold reached. No recovery needed.",
        }

        return reason_map.get(failure_type, "Agent selected the next best valid protocol.") + " " + change_text


class ValidatorAgent:
    """
    Checks that the protocol only uses valid values from the training data.
    """

    def __init__(self, oracle: BHOracle):
        self.oracle = oracle

        self.valid_values = {
            "ligand": set(oracle.train_df["ligand"].dropna().unique()),
            "additive": set(oracle.train_df["additive"].dropna().unique()),
            "base": set(oracle.train_df["base"].dropna().unique()),
            "aryl_halide": set(oracle.train_df["aryl_halide"].dropna().unique()),
        }

    def run(self, protocol: dict) -> dict:
        errors = []

        required_keys = ["ligand", "additive", "base", "aryl_halide"]

        for key in required_keys:
            if key not in protocol:
                errors.append(f"Missing key: {key}")
            elif protocol[key] not in self.valid_values[key]:
                errors.append(f"Invalid value for {key}: {protocol[key]}")

        if errors:
            return {
                "approved": False,
                "errors": errors,
            }

        return {
            "approved": True,
            "errors": [],
        }


class StopPolicyAgent:
    """
    Decides whether to stop or continue.
    """

    def run(self, history: list, success_threshold: float, max_attempts: int) -> dict:
        latest = history[-1]
        latest_yield = latest["result"]["yield_percent"]

        if latest_yield >= success_threshold:
            return {
                "stop": True,
                "reason": f"Success: yield reached {latest_yield}%, which is above the {success_threshold}% target.",
            }

        if len(history) >= max_attempts:
            return {
                "stop": True,
                "reason": f"Stopped: reached max attempts ({max_attempts}).",
            }

        return {
            "stop": False,
            "reason": "Continue: yield target not reached yet.",
        }


class CoreAgentLoop:
    """
    Phase 2 full loop:
    Planner -> Generator -> Simulator -> Classifier -> Recovery -> Validator -> Stop Policy
    """

    def __init__(self, csv_path: str):
        self.oracle = BHOracle(csv_path)
        self.planner = PlannerAgent()
        self.generator = ProtocolGeneratorAgent(self.oracle)
        self.simulator = SimulatorAgent(self.oracle)
        self.classifier = FailureClassifierAgent()
        self.recovery = RecoveryAgent(self.oracle)
        self.validator = ValidatorAgent(self.oracle)
        self.stop_policy = StopPolicyAgent()

    def run(self) -> dict:
        config = self.planner.run()

        print("=" * 70)
        print("PHASE 2: CORE AGENT LOOP")
        print("=" * 70)

        print("\nPlanner Agent Output")
        print("-" * 70)
        for key, value in config.items():
            print(f"{key}: {value}")

        generated = self.generator.run()
        current_protocol = generated["protocol"]

        print("\nProtocol Generator Output")
        print("-" * 70)
        print(generated["reason"])
        self.print_protocol(current_protocol)

        history = []
        previous_yield = None

        for attempt in range(1, config["max_attempts"] + 1):
            print("\n" + "=" * 70)
            print(f"ATTEMPT {attempt}")
            print("=" * 70)

            validation = self.validator.run(current_protocol)

            print("\nValidator Agent Output")
            print("-" * 70)
            print(f"approved: {validation['approved']}")

            if not validation["approved"]:
                print(f"errors: {validation['errors']}")
                break

            result = self.simulator.run(current_protocol)

            print("\nSimulator Agent Output")
            print("-" * 70)
            for key, value in result.items():
                print(f"{key}: {value}")

            failure_type = self.classifier.run(result, previous_yield)

            print("\nFailure Classifier Output")
            print("-" * 70)
            print(f"failure_type: {failure_type}")

            history.append(
                {
                    "attempt": attempt,
                    "protocol": current_protocol,
                    "result": result,
                    "failure_type": failure_type,
                }
            )

            stop = self.stop_policy.run(
                history=history,
                success_threshold=config["success_threshold"],
                max_attempts=config["max_attempts"],
            )

            print("\nStop Policy Output")
            print("-" * 70)
            print(f"stop: {stop['stop']}")
            print(f"reason: {stop['reason']}")

            if stop["stop"]:
                break

            tried_protocols = [item["protocol"] for item in history]

            recovery_output = self.recovery.run(
                current_protocol=current_protocol,
                failure_type=failure_type,
                tried_protocols=tried_protocols,
            )

            print("\nRecovery Agent Output")
            print("-" * 70)
            print(f"reason: {recovery_output['reason']}")
            print(f"expected_known_yield: {recovery_output['expected_known_yield']}")

            current_protocol = recovery_output["protocol_patch"]
            previous_yield = result["yield_percent"]

            print("\nNext Protocol")
            print("-" * 70)
            self.print_protocol(current_protocol)

        print("\n" + "=" * 70)
        print("PHASE 2 COMPLETE")
        print("=" * 70)

        print("\nFinal Iteration History")
        print("-" * 70)

        for item in history:
            print(
                f"Attempt {item['attempt']}: "
                f"yield={item['result']['yield_percent']}%, "
                f"failure_type={item['failure_type']}"
            )

        return {
            "config": config,
            "history": history,
        }

    def print_protocol(self, protocol: dict):
        print(f"ligand: {protocol['ligand']}")
        print(f"additive: {protocol['additive']}")
        print(f"base: {protocol['base']}")
        print(f"aryl_halide: {protocol['aryl_halide']}")


def main():
    loop = CoreAgentLoop(CSV_PATH)
    loop.run()


if __name__ == "__main__":
    main()