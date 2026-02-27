"""
Test suite: Data Pipeline, Preprocessing & Integration
Valida el flujo completo: datos → normalización → clustering → avatars → export.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import time


# ═══════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════

class TestDataGeneration:

    def test_generates_correct_number_of_rows(self, fake_customer_df):
        assert len(fake_customer_df) == 200

    def test_has_required_columns(self, fake_customer_df):
        required = ["user_key", "rfm_r_score", "rfm_f_score", "rfm_m_score",
                     "rfm_composite", "age", "engagement_score", "avg_session_duration_sec"]
        for col in required:
            assert col in fake_customer_df.columns, f"Falta columna: {col}"

    def test_numeric_ranges_valid(self, fake_customer_df):
        df = fake_customer_df
        assert df["rfm_r_score"].min() >= 0
        assert df["rfm_r_score"].max() <= 1
        assert df["rfm_f_score"].min() >= 0
        assert df["rfm_m_score"].min() >= 0
        assert df["age"].min() >= 18
        assert df["age"].max() <= 75
        assert df["engagement_score"].min() >= 1
        assert df["engagement_score"].max() <= 10

    def test_no_nulls_in_numerics(self, fake_customer_df, numeric_features):
        for f in numeric_features:
            assert fake_customer_df[f].isnull().sum() == 0, f"NaN en {f}"

    def test_categorical_values_valid(self, fake_customer_df):
        assert set(fake_customer_df["gender"].unique()) <= {"M", "F"}
        assert len(fake_customer_df["city"].unique()) > 1
        assert len(fake_customer_df["acquisition_channel"].unique()) > 1


# ═══════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════

class TestPreprocessing:

    def test_standard_scaler_zero_mean(self, fake_customer_df, numeric_features):
        X = fake_customer_df[numeric_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)

    def test_standard_scaler_unit_variance(self, fake_customer_df, numeric_features):
        X = fake_customer_df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=0.05)

    def test_minmax_scaler_range_01(self, fake_customer_df, numeric_features):
        X = fake_customer_df[numeric_features].values
        X_scaled = MinMaxScaler().fit_transform(X)
        assert X_scaled.min() >= 0.0
        assert X_scaled.max() <= 1.0

    def test_pca_reduces_dimensions(self, fake_customer_df, numeric_features):
        X = fake_customer_df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        assert X_2d.shape == (len(X), 2)

    def test_pca_variance_explained_sums_le_one(self, fake_customer_df, numeric_features):
        X = fake_customer_df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        total_var = pca.explained_variance_ratio_.sum()
        assert 0 < total_var <= 1.0

    def test_handles_constant_column(self):
        """Si una columna es constante, StandardScaler maneja con std=0."""
        X = np.array([[1, 5], [1, 10], [1, 15]])
        X_scaled = StandardScaler().fit_transform(X)
        # Columna constante se hace 0
        assert np.all(X_scaled[:, 0] == 0)


# ═══════════════════════════════════════════════════════
# END-TO-END PIPELINE
# ═══════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Testa el flujo completo: datos → scale → cluster → avatar → métricas."""

    @pytest.mark.parametrize("method_cls,params", [
        (KMeans, {"n_clusters": 4, "random_state": 42}),
        (MiniBatchKMeans, {"n_clusters": 4, "random_state": 42, "batch_size": 50}),
        (AgglomerativeClustering, {"n_clusters": 4, "linkage": "ward"}),
        (Birch, {"n_clusters": 4, "threshold": 0.5}),
    ])
    def test_full_pipeline_k_methods(self, fake_customer_df, numeric_features,
                                      method_cls, params):
        """Pipeline completo para métodos que requieren K."""
        df = fake_customer_df
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)

        # Clustering
        model = method_cls(**params)
        labels = model.fit_predict(X_scaled)

        # Validaciones básicas
        assert len(labels) == len(df)
        assert len(set(labels)) == 4
        assert all(l >= 0 for l in labels)

        # Métricas
        sil = silhouette_score(X_scaled, labels)
        assert -1 <= sil <= 1

        # Avatars (medoide)
        for cl in range(4):
            mask = labels == cl
            cluster_pts = X_scaled[mask]
            assert len(cluster_pts) > 0, f"Cluster {cl} vacío"
            dist_matrix = cdist(cluster_pts, cluster_pts)
            medoid_local = dist_matrix.sum(axis=1).argmin()
            assert 0 <= medoid_local < len(cluster_pts)

    def test_full_pipeline_dbscan(self, fake_customer_df, numeric_features):
        """Pipeline completo para DBSCAN."""
        X = fake_customer_df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)

        labels = DBSCAN(eps=2.0, min_samples=5).fit_predict(X_scaled)
        assert len(labels) == len(X_scaled)

        n_clusters = len(set(labels) - {-1})
        # DBSCAN puede encontrar 0+ clusters
        if n_clusters >= 2:
            valid = labels >= 0
            sil = silhouette_score(X_scaled[valid], labels[valid])
            assert -1 <= sil <= 1

    def test_pipeline_with_2_features(self, fake_customer_df):
        """Funciona con el mínimo de 2 features."""
        features = ["rfm_r_score", "rfm_m_score"]
        X = fake_customer_df[features].values
        X_scaled = StandardScaler().fit_transform(X)

        labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
        assert len(set(labels)) == 3

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        assert X_2d.shape[1] == 2

    def test_pipeline_with_all_features(self, fake_customer_df, numeric_features):
        """Funciona con todas las features numéricas."""
        X = fake_customer_df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)

        labels = KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled)
        assert len(set(labels)) == 5

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        assert X_2d.shape == (len(X), 2)


# ═══════════════════════════════════════════════════════
# EXPORT / OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════

class TestExportOutput:

    def test_cluster_column_added(self, fake_customer_df, numeric_features):
        df = fake_customer_df.copy()
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        df["cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
        assert "cluster" in df.columns
        assert df["cluster"].isnull().sum() == 0

    def test_avatar_dataframe_structure(self, fake_customer_df, numeric_features):
        """El DataFrame de avatars tiene la estructura esperada."""
        df = fake_customer_df
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

        avatars = []
        for cl in sorted(set(labels)):
            mask = labels == cl
            cluster_df = df[mask]
            cluster_pts = X_scaled[mask]
            dist_matrix = cdist(cluster_pts, cluster_pts)
            medoid_local = dist_matrix.sum(axis=1).argmin()
            rep = cluster_df.iloc[medoid_local]
            avatars.append({
                "cluster": cl,
                "size": int(mask.sum()),
                "pct": round(mask.sum() / len(labels) * 100, 1),
                "avatar_name": rep.get("avatar_name", ""),
            })

        avatar_df = pd.DataFrame(avatars)
        assert len(avatar_df) == 3
        assert "cluster" in avatar_df.columns
        assert "size" in avatar_df.columns
        assert "pct" in avatar_df.columns
        assert avatar_df["size"].sum() == len(df)
        assert abs(avatar_df["pct"].sum() - 100.0) < 1.0

    def test_csv_export_roundtrip(self, fake_customer_df, numeric_features, tmp_path):
        """CSV export → reimport mantiene integridad."""
        df = fake_customer_df.copy()
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        df["cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

        csv_path = tmp_path / "output.csv"
        df.to_csv(csv_path, index=False)
        df_reload = pd.read_csv(csv_path)

        assert len(df_reload) == len(df)
        assert "cluster" in df_reload.columns
        assert set(df_reload["cluster"].unique()) == {0, 1, 2}


# ═══════════════════════════════════════════════════════
# SCALABILITY
# ═══════════════════════════════════════════════════════

class TestScalability:

    @pytest.mark.parametrize("n", [500, 1000, 5000])
    def test_kmeans_scales(self, n):
        """K-Means++ debe correr en <2s para n ≤ 5000."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, 6))
        X_scaled = StandardScaler().fit_transform(X)

        t0 = time.time()
        labels = KMeans(n_clusters=5, random_state=42, n_init=5).fit_predict(X_scaled)
        elapsed = time.time() - t0

        assert len(set(labels)) == 5
        assert elapsed < 2.0, f"K-Means en n={n} tardó {elapsed:.2f}s (>2s)"

    @pytest.mark.parametrize("n", [500, 1000, 5000])
    def test_minibatch_scales(self, n):
        """Mini-Batch K-Means debe ser más rápido que K-Means standard."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, 6))
        X_scaled = StandardScaler().fit_transform(X)

        t0 = time.time()
        MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=500).fit_predict(X_scaled)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Mini-Batch en n={n} tardó {elapsed:.2f}s (>1s)"

    def test_birch_linear_scaling(self):
        """BIRCH debe escalar linealmente (Zhang 1996)."""
        rng = np.random.default_rng(42)
        times = []
        for n in [1000, 2000, 4000]:
            X = rng.normal(size=(n, 4))
            t0 = time.time()
            Birch(n_clusters=5, threshold=0.5).fit_predict(X)
            times.append(time.time() - t0)

        # Ratio de tiempo 4000/1000 debería ser ~4x (lineal), no 16x (cuadrático)
        ratio = times[2] / max(times[0], 0.001)
        assert ratio < 10, f"BIRCH scaling ratio {ratio:.1f}x no es lineal"


# ═══════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════

class TestEdgeCases:

    def test_k_equals_n(self):
        """K = número de puntos → cada punto es su propio cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = KMeans(n_clusters=3, random_state=42, n_init=1).fit_predict(X)
        assert len(set(labels)) == 3

    def test_identical_points(self):
        """Todos los puntos idénticos → no debería crashear."""
        X = np.ones((50, 3))
        labels = KMeans(n_clusters=2, random_state=42, n_init=1).fit_predict(X)
        assert len(labels) == 50

    def test_single_feature(self):
        """1D clustering debe funcionar."""
        X = np.array([[1], [2], [100], [101]])
        labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)
        assert len(set(labels)) == 2

    def test_many_dimensions(self):
        """Alta dimensionalidad (curse of dimensionality)."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 50))
        labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        assert len(set(labels)) == 3

    def test_dbscan_all_noise(self):
        """DBSCAN con parámetros estrictos → todo noise."""
        X = np.random.randn(50, 2) * 10
        labels = DBSCAN(eps=0.001, min_samples=10).fit_predict(X)
        assert all(l == -1 for l in labels)

    def test_empty_cluster_handling(self):
        """Si un cluster queda vacío, sklearn lo maneja."""
        # Con K muy grande y pocos puntos
        X = np.array([[0, 0], [1, 0]])
        # K=2 con 2 puntos → cada uno su cluster
        labels = KMeans(n_clusters=2, random_state=42, n_init=1).fit_predict(X)
        assert len(set(labels)) == 2


# ═══════════════════════════════════════════════════════
# OPTIMAL K SEARCH
# ═══════════════════════════════════════════════════════

class TestOptimalK:

    def test_elbow_inertia_decreases(self, scaled_data):
        """Inercia debe decrecer monótonamente al aumentar K."""
        X, _ = scaled_data
        inertias = []
        for k in range(2, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X)
            inertias.append(km.inertia_)
        # Monótonamente decreciente
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1] - 1e-10

    def test_silhouette_finds_true_k(self, well_separated_clusters):
        """Silhouette máximo debería coincidir con el K verdadero (k=3)."""
        X, _ = well_separated_clusters
        X_scaled = StandardScaler().fit_transform(X)
        sils = {}
        for k in range(2, 7):
            labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(X_scaled)
            sils[k] = silhouette_score(X_scaled, labels)
        best_k = max(sils, key=sils.get)
        assert best_k == 3, f"Optimal K={best_k}, esperado K=3. Sils: {sils}"
