"""
Shared fixtures for Clustering Persona Engine tests.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def well_separated_clusters():
    """3 clusters gaussianos bien separados para tests deterministas."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0, 0], scale=0.3, size=(50, 3))
    c2 = rng.normal(loc=[6, 6, 6], scale=0.3, size=(50, 3))
    c3 = rng.normal(loc=[12, 0, 0], scale=0.3, size=(50, 3))
    X = np.vstack([c1, c2, c3])
    true_labels = np.array([0]*50 + [1]*50 + [2]*50)
    return X, true_labels


@pytest.fixture
def overlapping_clusters():
    """Clusters con overlap parcial para tests de robustez."""
    rng = np.random.default_rng(123)
    c1 = rng.normal(loc=[0, 0], scale=1.0, size=(100, 2))
    c2 = rng.normal(loc=[3, 3], scale=1.0, size=(100, 2))
    X = np.vstack([c1, c2])
    true_labels = np.array([0]*100 + [1]*100)
    return X, true_labels


@pytest.fixture
def scaled_data(well_separated_clusters):
    """Datos escalados con StandardScaler."""
    X, labels = well_separated_clusters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, labels


@pytest.fixture
def fake_customer_df():
    """DataFrame fake simulando dim_user_behavior (Variables_Cruce_BD_Leo_v1.md §2.1)."""
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "user_key": list(range(1, n + 1)),
        "business_key": rng.integers(1, 101, n),
        "avatar_name": [f"Client_{i}" for i in range(n)],
        "rfm_r_score": np.clip(rng.normal(0.5, 0.25, n), 0, 1).round(3),
        "rfm_f_score": np.clip(rng.normal(0.5, 0.25, n), 0, 1).round(3),
        "rfm_m_score": np.clip(rng.normal(0.5, 0.25, n), 0, 1).round(3),
        "rfm_composite": None,  # computed below
        "total_interactions": np.clip(rng.normal(15, 10, n), 1, 80).astype(int),
        "total_spend_cents": np.clip(rng.normal(30000, 20000, n), 500, 200000).astype(int),
        "age": np.clip(rng.normal(35, 10, n), 18, 75).astype(int),
        "engagement_score": np.clip(rng.normal(6, 2, n), 1, 10).round(1),
        "avg_session_duration_sec": np.clip(rng.normal(900, 480, n), 30, 3600).round(1),
        "city": rng.choice(["Bogotá", "Medellín", "Lima", "CDMX"], n),
        "country_code": rng.choice(["CO", "CO", "PE", "MX"], n),
        "acquisition_channel": rng.choice(["Email", "Social", "Organic", "Paid"], n),
        "device_type": rng.choice(["Mobile", "Desktop", "Tablet"], n),
        "income_bracket": rng.choice(["<30K", "30K-60K", "60K-100K", "100K+"], n),
        "gender": rng.choice(["M", "F"], n),
    }
    df = pd.DataFrame(data)
    df["rfm_composite"] = ((df["rfm_r_score"] + df["rfm_f_score"] + df["rfm_m_score"]) / 3).round(3)
    return df


@pytest.fixture
def numeric_features():
    return ["rfm_r_score", "rfm_f_score", "rfm_m_score", "rfm_composite",
            "age", "engagement_score", "avg_session_duration_sec"]


@pytest.fixture
def large_dataset():
    """Dataset de 5000 puntos para tests de escalabilidad."""
    rng = np.random.default_rng(42)
    centers = rng.uniform(-10, 10, size=(5, 4))
    X = np.vstack([rng.normal(loc=c, scale=0.8, size=(1000, 4)) for c in centers])
    return X
