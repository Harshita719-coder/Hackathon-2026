import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = Path("/mnt/data/buchwald_hartwig_huggingface.csv")
OUT_DIR = Path("/mnt/data/bh_outputs")
OUT_DIR.mkdir(exist_ok=True)

COMPONENT_COLS = ["ligand", "additive", "base", "aryl_halide", "product_string"]
CAT_COLS = ["ligand", "additive", "base", "aryl_halide"]


def morgan_bits(smiles: str, radius: int = 2, nbits: int = 256) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings": rdMolDescriptors.CalcNumRings,
    "FracCSP3": rdMolDescriptors.CalcFractionCSP3,
    "LogP": Descriptors.MolLogP,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
}


def descriptor_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return np.array([func(mol) for func in DESCRIPTOR_FUNCS.values()], dtype=np.float32)



def build_features(df: pd.DataFrame):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_cat = ohe.fit_transform(df[CAT_COLS])

    x_morgan = sp.csr_matrix(
        np.hstack([
            np.vstack(df[col].map(lambda s: morgan_bits(s, 2, 256)).values)
            for col in COMPONENT_COLS
        ])
    )

    desc_blocks = [np.vstack(df[col].map(descriptor_vector).values) for col in COMPONENT_COLS]
    desc_stack = np.stack(desc_blocks, axis=1)
    x_desc = np.hstack(desc_blocks + [desc_stack.sum(axis=1), desc_stack.mean(axis=1), desc_stack.std(axis=1)])
    x_hybrid = sp.hstack([x_morgan, sp.csr_matrix(x_desc)], format="csr")

    return {
        "categorical_ohe": x_cat,
        "morgan_concat": x_morgan,
        "descriptors": x_desc,
        "hybrid_morgan_desc": x_hybrid,
    }



def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }



def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["yield_clean"] = df["yield"].clip(0, 100)
    df = df.drop_duplicates(subset=["canonical_rxn_smiles"]).copy()

    features = build_features(df)
    y = df["yield_clean"].to_numpy()

    train_idx = df.index[df["split"] == "train"].to_numpy()
    valid_idx = df.index[df["split"] == "valid"].to_numpy()
    test_idx = df.index[df["split"] == "test"].to_numpy()

    groups = df["aryl_halide"].to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    _, temp_idx = next(gss1.split(df, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    valid_rel, test_rel = next(gss2.split(df.iloc[temp_idx], y[temp_idx], groups=groups[temp_idx]))
    group_test_idx = temp_idx[test_rel]

    candidate_models = {
        ("categorical_ohe", "RF"): RandomForestRegressor(
            n_estimators=200, max_depth=20, max_features="sqrt", n_jobs=-1, random_state=42
        ),
        ("morgan_concat", "RF"): RandomForestRegressor(
            n_estimators=250, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=42
        ),
        ("descriptors", "XGB"): XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, reg_lambda=2, objective="reg:squarederror",
            tree_method="hist", n_jobs=4, random_state=42,
        ),
        ("descriptors", "SVR"): Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=10, gamma="scale", epsilon=1.0)),
        ]),
        ("hybrid_morgan_desc", "XGB"): XGBRegressor(
            n_estimators=180, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.7, reg_lambda=2, objective="reg:squarederror",
            tree_method="hist", n_jobs=4, random_state=42,
        ),
    }

    rows = []
    trained = {}
    for (feature_name, model_name), model in candidate_models.items():
        x = features[feature_name]
        model.fit(x[train_idx], y[train_idx])
        rows.append({
            "feature_set": feature_name,
            "model": model_name,
            **{f"valid_{k}": v for k, v in metrics(y[valid_idx], model.predict(x[valid_idx])).items()},
            **{f"test_{k}": v for k, v in metrics(y[test_idx], model.predict(x[test_idx])).items()},
            **{f"group_test_{k}": v for k, v in metrics(y[group_test_idx], model.predict(x[group_test_idx])).items()},
        })
        trained[(feature_name, model_name)] = model

    results = pd.DataFrame(rows).sort_values("valid_RMSE")
    results.to_csv(OUT_DIR / "model_results.csv", index=False)

    model_a = trained[("hybrid_morgan_desc", "XGB")]
    model_b = trained[("morgan_concat", "RF")]
    ensemble_valid = 0.5 * model_a.predict(features["hybrid_morgan_desc"][valid_idx]) + 0.5 * model_b.predict(features["morgan_concat"][valid_idx])
    ensemble_test = 0.5 * model_a.predict(features["hybrid_morgan_desc"][test_idx]) + 0.5 * model_b.predict(features["morgan_concat"][test_idx])
    ensemble_group_test = 0.5 * model_a.predict(features["hybrid_morgan_desc"][group_test_idx]) + 0.5 * model_b.predict(features["morgan_concat"][group_test_idx])

    summary = {
        "rows": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "yield_summary": df["yield_clean"].describe().to_dict(),
        "ensemble": {
            "valid": metrics(y[valid_idx], ensemble_valid),
            "test": metrics(y[test_idx], ensemble_test),
            "group_test": metrics(y[group_test_idx], ensemble_group_test),
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(results.to_string(index=False))
    print("\nEnsemble test:", summary["ensemble"]["test"])


if __name__ == "__main__":
    main()
