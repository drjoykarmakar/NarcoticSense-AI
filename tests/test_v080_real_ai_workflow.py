import numpy as np
import pandas as pd

from narcoticsense.models import (
    list_model_registry,
    load_real_ai_model,
    predict_unknowns_with_bundle,
    prepare_training_table,
    train_save_real_ai_classifier,
    validate_training_table,
)
from narcoticsense.spectroscopy.dataset import AlignedDataset


def _aligned_dataset():
    rng = np.random.default_rng(80)
    x_axis = np.linspace(200, 700, 40)
    class_a = rng.normal(0.0, 0.10, size=(6, 40))
    class_b = rng.normal(2.0, 0.10, size=(6, 40))
    unknown = rng.normal(0.05, 0.10, size=(2, 40))
    matrix = np.vstack([class_a, class_b, unknown])
    sample_ids = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)] + ["UNK0", "UNK1"]
    return AlignedDataset(
        x=x_axis,
        matrix=matrix,
        sample_ids=sample_ids,
        modalities=["fluorescence"] * len(sample_ids),
        metadata=[{} for _ in sample_ids],
    )


def _metadata(sample_ids):
    labels = ["class_a"] * 6 + ["class_b"] * 6 + ["unknown", ""]
    return pd.DataFrame({"sample_id": sample_ids, "compound_or_class": labels})


def test_prepare_and_validate_training_table():
    aligned = _aligned_dataset()
    metadata = _metadata(aligned.sample_ids)
    table = prepare_training_table(aligned.sample_ids, metadata)
    assert table["training_status"].tolist().count("labeled") == 12
    assert table["training_status"].tolist().count("unknown") == 2
    checks = validate_training_table(table)
    assert "number of classes" in checks["check"].tolist()
    assert checks[checks["status"] == "fail"].empty


def test_train_save_registry_and_predict_unknowns(tmp_path):
    aligned = _aligned_dataset()
    metadata = _metadata(aligned.sample_ids)
    result = train_save_real_ai_classifier(
        aligned,
        metadata,
        tmp_path,
        model_name="Random Forest",
        project_name="Unit Test Project",
    )
    assert result.model_path.exists()
    registry = list_model_registry(tmp_path)
    assert registry.shape[0] == 1
    assert registry["task"].iloc[0] == "classification"
    bundle = load_real_ai_model(result.model_path)
    prediction = predict_unknowns_with_bundle(bundle, aligned, confidence_threshold=0.5)
    assert prediction.predictions.shape[0] == aligned.n_samples
    assert "decision_flag" in prediction.predictions.columns
    assert "predicted_label" in prediction.predictions.columns
