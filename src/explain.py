"""
explain.py

SHAP attributions for the ML correction, so the hybrid model can say WHY
it adjusted the physics baseline rather than just that it did.

WHAT IS BEING EXPLAINED, PRECISELY
==================================
Not "why is this iceberg going north-east" -- that is mostly the ocean
current, and the physics term already answers it. What SHAP explains here
is the RESIDUAL: the correction the model applies on top of calibrated
free drift. So an attribution reads as "the model pushed this iceberg
0.012 m/s further east than free drift predicted, and the reason was
mostly the previous segment's eastward residual".

That framing matters when reading the numbers. A feature with a large
attribution is not a large driver of the iceberg's motion; it is a large
driver of the model's DISAGREEMENT with the physics. Those are different
claims, and conflating them overstates what the ML stage is doing.

WHY TreeExplainer
=================
The residual models are gradient-boosted trees, so exact Shapley values
are computable in polynomial time (TreeSHAP) rather than estimated by
sampling. No background dataset or approximation error, and it runs in
milliseconds per row -- fast enough to explain a forecast live in the
dashboard while the operator is looking at it.

Attributions are in the model's own units, m/s, and sum exactly to
(prediction - base_value). That additivity is checked in the demo below,
because an explanation that does not reconstruct the prediction is not an
explanation of that prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Human-readable names for the feature columns, used when an explanation
# is rendered for someone who did not write the feature table.
FEATURE_LABELS: dict[str, str] = {
    "phys_u": "physics baseline (east)",
    "phys_v": "physics baseline (north)",
    "u_wind": "wind, eastward",
    "v_wind": "wind, northward",
    "u_current": "current, eastward",
    "v_current": "current, northward",
    "wind_speed": "wind speed",
    "current_speed": "current speed",
    "lat": "latitude",
    "lon_sin": "longitude (sin)",
    "lon_cos": "longitude (cos)",
    "log_area": "iceberg size",
    "dt_hours": "segment length",
}


def _label(column: str) -> str:
    """Render a feature column name in human-readable form.

    Args:
        column: A feature column name, possibly with a "_t-N" lag suffix.

    Returns:
        A readable label, e.g. "residual_u_t-2" -> "past residual (east), 2 steps back".
    """
    if column in FEATURE_LABELS:
        return FEATURE_LABELS[column]
    if "_t-" in column:
        base, lag = column.rsplit("_t-", 1)
        base_labels = {
            "obs_u": "past observed velocity (east)",
            "obs_v": "past observed velocity (north)",
            "residual_u": "past residual (east)",
            "residual_v": "past residual (north)",
            "u_wind": "past wind, eastward",
            "v_wind": "past wind, northward",
            "u_current": "past current, eastward",
            "v_current": "past current, northward",
        }
        readable = base_labels.get(base, base)
        return f"{readable}, {lag} step{'s' if lag != '1' else ''} back"
    return column


class ResidualExplainer:
    """Wraps TreeSHAP explainers for the residual_u and residual_v models.

    Built once from a trained bundle and then reused: constructing a
    TreeExplainer parses the whole tree ensemble, which is far more
    expensive than explaining an individual row.

    Attributes:
        feature_cols: The feature order the underlying models expect.
    """

    def __init__(self, models: dict) -> None:
        """Build TreeSHAP explainers for both residual components.

        Args:
            models: A bundle from train_model.train_residual_model().

        Raises:
            RuntimeError: If the shap package is missing.
            TypeError: If the bundle holds estimators TreeSHAP cannot
                explain (e.g. the ridge variant), naming the model type.
        """
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError(
                "ResidualExplainer: the 'shap' package is not installed. Install it with "
                "`pip install shap`."
            ) from exc

        model_type = models.get("model_type", "xgb")
        if model_type != "xgb":
            raise TypeError(
                f"ResidualExplainer: TreeSHAP explains tree ensembles, but this bundle "
                f"holds a '{model_type}' model. Train with model_type='xgb' to explain it "
                f"(a linear model's coefficients are already its explanation)."
            )

        self.feature_cols: list[str] = list(models["feature_cols"])
        self._explainers = {
            "u": shap.TreeExplainer(models["u"]),
            "v": shap.TreeExplainer(models["v"]),
        }

    def explain_row(
        self, feature_row: pd.DataFrame | pd.Series, component: str = "u"
    ) -> pd.DataFrame:
        """Attribute one residual prediction to its features.

        Args:
            feature_row: A single feature row, as a Series or a one-row
                DataFrame.
            component: "u" (eastward) or "v" (northward).

        Returns:
            A DataFrame with feature, label, value, shap_value and
            abs_shap, sorted by absolute contribution descending. The
            shap_value column is in m/s and sums with the base value to
            the model's prediction.

        Raises:
            ValueError: If component is not "u" or "v".
            KeyError: If the row is missing any expected feature.
        """
        if component not in self._explainers:
            raise ValueError(
                f"explain_row: component must be 'u' or 'v', got {component!r}."
            )
        frame = (
            feature_row.to_frame().T if isinstance(feature_row, pd.Series) else feature_row
        )
        missing = [c for c in self.feature_cols if c not in frame.columns]
        if missing:
            raise KeyError(f"explain_row: feature row is missing {missing}.")
        frame = frame[self.feature_cols].astype(float)

        values = np.asarray(self._explainers[component].shap_values(frame)).reshape(-1)
        table = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "label": [_label(c) for c in self.feature_cols],
                "value": frame.iloc[0].to_numpy(dtype=float),
                "shap_value": values,
            }
        )
        table["abs_shap"] = table["shap_value"].abs()
        return table.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    def base_value(self, component: str = "u") -> float:
        """Return the explainer's base (expected) value for a component.

        This is the model's average output over its training data -- the
        prediction it would make knowing nothing about a specific row.
        SHAP values are deviations from it.

        Args:
            component: "u" or "v".

        Returns:
            The base value in m/s.
        """
        return float(np.ravel(self._explainers[component].expected_value)[0])

    def narrate(
        self, feature_row: pd.DataFrame | pd.Series, top_n: int = 3
    ) -> str:
        """Describe in one sentence why the model corrected the physics baseline.

        Written for the dashboard, where an operator needs the gist in a
        glance and the full table only if the gist looks wrong. The
        correction is reported as a speed and compass direction, because
        "3.4 km/day toward the north-east" is checkable against a map in
        a way that "residual_u = 0.021, residual_v = 0.018" is not.

        Args:
            feature_row: A single feature row.
            top_n: How many drivers to name.

        Returns:
            A plain-text sentence.
        """
        parts: list[str] = []
        total = {}
        for component in ("u", "v"):
            table = self.explain_row(feature_row, component)
            total[component] = self.base_value(component) + table["shap_value"].sum()
            parts.append(table)

        speed_ms = float(np.hypot(total["u"], total["v"]))
        km_per_day = speed_ms * 86.4  # m/s -> km/day
        bearing = (np.degrees(np.arctan2(total["u"], total["v"])) + 360.0) % 360.0
        compass = ["north", "north-east", "east", "south-east",
                   "south", "south-west", "west", "north-west"][int((bearing + 22.5) // 45) % 8]

        # Rank drivers by their combined influence on both components.
        combined = (
            pd.concat(parts)
            .groupby(["feature", "label"], as_index=False)["abs_shap"]
            .sum()
            .sort_values("abs_shap", ascending=False)
        )
        drivers = ", ".join(combined["label"].head(top_n))

        if km_per_day < 0.1:
            return f"The model left the physics forecast essentially unchanged (<0.1 km/day). Main influences: {drivers}."
        return (
            f"The model shifted the physics forecast by {km_per_day:.1f} km/day toward the "
            f"{compass}. This correction was driven mainly by {drivers}."
        )


def global_importance(
    models: dict, feature_df: pd.DataFrame, max_rows: int = 500
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across a sample of rows.

    Mean |SHAP| is a better global ranking than XGBoost's built-in
    `feature_importances_`: the built-in counts how often a feature was
    split on and by how much it reduced training loss, which inflates
    high-cardinality features regardless of whether they moved
    predictions. Mean |SHAP| measures actual effect on the output, in m/s.

    Args:
        models: A trained bundle.
        feature_df: Rows to explain; sampled if larger than max_rows.
        max_rows: Cap on rows explained, for speed.

    Returns:
        A DataFrame with feature, label, mean_abs_shap_u,
        mean_abs_shap_v and mean_abs_shap (their sum), sorted
        descending.
    """
    explainer = ResidualExplainer(models)
    frame = feature_df[explainer.feature_cols].astype(float)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=42)

    columns = {}
    for component in ("u", "v"):
        values = np.asarray(explainer._explainers[component].shap_values(frame))
        columns[f"mean_abs_shap_{component}"] = np.abs(values).mean(axis=0)

    table = pd.DataFrame(
        {
            "feature": explainer.feature_cols,
            "label": [_label(c) for c in explainer.feature_cols],
            **columns,
        }
    )
    table["mean_abs_shap"] = table["mean_abs_shap_u"] + table["mean_abs_shap_v"]
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_ingest import build_real_dataset
    from features import build_feature_table, calibrate_drift_params, compute_observed_velocity
    from train_model import predict_residual, train_residual_model

    pooled, _summary = build_real_dataset(verbose=False)
    params = calibrate_drift_params(compute_observed_velocity(pooled))
    feature_df, feature_cols, target_cols = build_feature_table(pooled, params=params)
    models = train_residual_model(feature_df, feature_cols, target_cols)

    explainer = ResidualExplainer(models)
    row = feature_df.iloc[[0]]

    print("Global feature importance (mean |SHAP|, m/s):")
    print(global_importance(models, feature_df).head(8).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\nExplaining one forecast for iceberg {feature_df['iceberg_id'].iloc[0]}:")
    for component in ("u", "v"):
        table = explainer.explain_row(row, component)
        predicted = explainer.base_value(component) + table["shap_value"].sum()
        actual = predict_residual(models, row)[0 if component == "u" else 1]
        # Additivity: SHAP values must reconstruct the model's own
        # prediction exactly, or they are not explaining this prediction.
        assert abs(predicted - actual) < 1e-4, (
            f"SHAP additivity violated for residual_{component}: "
            f"base + sum(shap) = {predicted:.6f} but the model predicted {actual:.6f}"
        )
        print(f"\n  residual_{component} = {actual:+.5f} m/s "
              f"(base {explainer.base_value(component):+.5f} + contributions)")
        print(table.head(5)[["label", "value", "shap_value"]].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    print(f"\nNarrative: {explainer.narrate(row)}")
    print("\nexplain.py checks passed (SHAP values reconstruct the predictions exactly).")
