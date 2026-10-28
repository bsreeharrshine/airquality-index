from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌿",
    layout="wide",
)

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


# -------------------------------------------------
# GENERATE SAMPLE DATA
# -------------------------------------------------
def generate_sample_data():
    """Synthetic data for demonstrating the ML project."""
    rng = np.random.default_rng(42)
    rows = 2400

    pollution = rng.uniform(0, 1, rows)

    scales = np.array([300, 500, 220, 160, 12, 200])

    readings = np.maximum(
        0,
        pollution[:, None]
        * scales
        * rng.lognormal(0, 0.22, size=(rows, 6)),
    )

    data = pd.DataFrame(readings.round(2), columns=FEATURES)

    # Simplified teaching relationships, not an official AQI calculator.
    pm25_index = np.interp(
        data["PM2.5"],
        [0, 30, 60, 90, 120, 250, 400],
        [0, 50, 100, 200, 300, 400, 500],
    )

    pm10_index = np.interp(
        data["PM10"],
        [0, 50, 100, 250, 350, 430, 600],
        [0, 50, 100, 200, 300, 400, 500],
    )

    no2_index = np.interp(
        data["NO2"],
        [0, 40, 80, 180, 280, 400, 600],
        [0, 50, 100, 200, 300, 400, 500],
    )

    target = np.maximum.reduce(
        [pm25_index, pm10_index, no2_index]
    )

    data["AQI"] = np.clip(
        target + rng.normal(0, 8, rows),
        0,
        500,
    ).round(1)

    return data


# -------------------------------------------------
# VALIDATE DATA
# -------------------------------------------------
def validate_data(data):
    required = FEATURES + ["AQI"]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if np.isinf(data.to_numpy(dtype=float)).any():
        raise ValueError("Dataset contains infinite values.")

    if (data < 0).any().any():
        raise ValueError("Values cannot be negative.")

    data = data.dropna(subset=["AQI"]).drop_duplicates()

    if len(data) < 50:
        raise ValueError("At least 50 distinct labeled rows are required.")

    if data["AQI"].nunique() < 2:
        raise ValueError("AQI values must vary.")

    if (data["AQI"] > 500).any():
        raise ValueError("AQI labels must be between 0 and 500.")

    if data[FEATURES].isna().all().any():
        raise ValueError("A pollutant column contains only missing values.")

    return data


# -------------------------------------------------
# AQI CATEGORY
# -------------------------------------------------
def get_category(aqi):
    rounded_aqi = int(np.rint(aqi))

    if rounded_aqi <= 50:
        return "Good"
    elif rounded_aqi <= 100:
        return "Satisfactory"
    elif rounded_aqi <= 200:
        return "Moderate"
    elif rounded_aqi <= 300:
        return "Poor"
    elif rounded_aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R²": r2_score(actual, predicted),
    }


# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
@st.cache_resource
def train_models(csv_bytes):
    if csv_bytes is None:
        data = generate_sample_data()
    else:
        data = pd.read_csv(BytesIO(csv_bytes))

    data = validate_data(data)

    X = data[FEATURES]
    y = data["AQI"]

    # Reserve 20% for final testing.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Total split: 60% training, 20% validation, 20% test.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42
    )

    candidates = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}
    validation_results = {}

    for name, estimator in candidates.items():
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            estimator,
        )

        pipeline.fit(X_train, y_train)

        validation_predictions = pipeline.predict(X_val)

        validation_results[name] = evaluate(
            y_val, validation_predictions
        )

        pipelines[name] = pipeline

    # Choose using validation data, not the final test set.
    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["MAE"],
    )

    best_model = pipelines[best_name]

    # Refit the selected model using the development data.
    best_model.fit(X_dev, y_dev)

    test_predictions = best_model.predict(X_test)
    test_metrics = evaluate(y_test, test_predictions)

    comparison = pd.DataFrame(validation_results).T

    test_results = pd.DataFrame(
        {
            "Actual AQI": y_test.to_numpy(),
            "Predicted AQI": test_predictions,
        }
    )

    ranges = {
        feature: (
            float(X_dev[feature].min()),
            float(X_dev[feature].max()),
        )
        for feature in FEATURES
    }

    return (
        data,
        best_model,
        best_name,
        comparison,
        test_metrics,
        test_results,
        ranges,
    )


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Project Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (optional)",
    type=["csv"],
)

st.sidebar.markdown(
    """
Required CSV columns:

`PM2.5, PM10, NO2, SO2, CO, O3, AQI`

Without an upload, the application uses synthetic sample data.
"""
)

csv_bytes = uploaded_file.getvalue() if uploaded_file else None


# -------------------------------------------------
# TRAIN / LOAD CACHED MODEL
# -------------------------------------------------
st.title("🌿 Air Quality Prediction System")
st.write("An end-to-end machine learning laboratory project.")

if uploaded_file is None:
    st.warning(
        "Demo mode: the dataset is synthetic. "
        "Evaluation scores do not represent real-world accuracy."
    )

try:
    with st.spinner("Preparing data and training models..."):
        (
            data,
            model,
            best_name,
            comparison,
            test_metrics,
            test_results,
            ranges,
        ) = train_models(csv_bytes)

except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()


# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Rows", len(data))
col2.metric("Selected Model", best_name)
col3.metric("Test MAE", f"{test_metrics['MAE']:.2f}")

st.caption(
    "The model estimates current AQI from pollutant measurements. "
    "It does not predict tomorrow's air quality."
)

predict_tab, data_tab, evaluation_tab = st.tabs(
    ["Predict AQI", "Dataset", "Model Evaluation"]
)


# -------------------------------------------------
# PREDICTION FORM
# -------------------------------------------------
with predict_tab:
    st.subheader("Enter Pollutant Measurements")

    st.caption(
        "Use µg/m³ for all pollutants except CO, which uses mg/m³. "
        "Use 24-hour averages for PM2.5, PM10, NO2 and SO2; "
        "8-hour averages for CO and O3."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            pm25 = st.number_input(
                "PM2.5 (µg/m³)", min_value=0.0, value=65.0
            )
            pm10 = st.number_input(
                "PM10 (µg/m³)", min_value=0.0, value=120.0
            )
            no2 = st.number_input(
                "NO2 (µg/m³)", min_value=0.0, value=45.0
            )

        with right:
            so2 = st.number_input(
                "SO2 (µg/m³)", min_value=0.0, value=20.0
            )
            co = st.number_input(
                "CO (mg/m³)", min_value=0.0, value=1.2
            )
            o3 = st.number_input(
                "O3 (µg/m³)", min_value=0.0, value=55.0
            )

        submitted = st.form_submit_button(
            "Predict Air Quality",
            type="primary",
        )

    if submitted:
        input_data = pd.DataFrame(
            [[pm25, pm10, no2, so2, co, o3]],
            columns=FEATURES,
        )

        if not np.isfinite(input_data.to_numpy()).all():
            st.error("Please enter finite numeric values.")
        else:
            prediction = float(model.predict(input_data)[0])
            category = get_category(prediction)

            result1, result2 = st.columns(2)

            result1.metric("Predicted AQI", f"{prediction:.1f}")
            result2.metric("Air Quality Category", category)

            outside_range = [
                feature
                for feature in FEATURES
                if not (
                    ranges[feature][0]
                    <= input_data.iloc[0][feature]
                    <= ranges[feature][1]
                )
            ]

            if outside_range:
                st.warning(
                    "Outside the training range: "
                    + ", ".join(outside_range)
                    + ". The prediction may be unreliable."
                )

            if prediction < 0 or prediction > 500:
                st.warning(
                    "Prediction is outside the expected AQI scale. "
                    "Treat this result as unreliable."
                )

    st.markdown(
        """
**AQI categories**

| AQI | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |
"""
    )


# -------------------------------------------------
# DATASET TAB
# -------------------------------------------------
with data_tab:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(data.describe(), use_container_width=True)

    st.download_button(
        "Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="air_quality_dataset.csv",
        mime="text/csv",
    )


# -------------------------------------------------
# EVALUATION TAB
# -------------------------------------------------
with evaluation_tab:
    st.subheader("Validation Set: Model Comparison")
    st.dataframe(comparison.round(3), use_container_width=True)

    st.bar_chart(comparison[["MAE"]])

    st.subheader("Selected Model: Held-out Test Results")

    a, b, c = st.columns(3)

    a.metric("MAE", f"{test_metrics['MAE']:.3f}")
    b.metric("RMSE", f"{test_metrics['RMSE']:.3f}")
    c.metric("R² Score", f"{test_metrics['R²']:.4f}")

    st.subheader("Actual vs Predicted AQI")
    st.scatter_chart(
        test_results,
        x="Actual AQI",
        y="Predicted AQI",
    )

    st.dataframe(
        test_results.head(30).round(2),
        use_container_width=True,
    )

    st.download_button(
        "Download Test Predictions",
        data=test_results.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )

    st.info(
        "MAE and RMSE measure prediction error in AQI points; "
        "lower is better. R² measures fit; higher is better. "
        "The model is selected using validation MAE, then evaluated "
        "on a separate test set."
    )

    st.caption(
        "For real station time-series data, use chronological or "
        "station-grouped splits instead of this demonstration's random split."
    )