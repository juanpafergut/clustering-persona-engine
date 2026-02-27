"""
Test suite: Avatar Methods & Quality Metrics
Valida representantes de cluster y métricas de calidad.

Papers:
  - Centroide (MacQueen, Berkeley 1967)
  - Medoide / PAM (Kaufman & Rousseeuw, 1990)
  - Nearest-to-Centroid (Bien & Tibshirani, JASA 2011)
  - Silhouette (Rousseeuw, 1987)
  - Calinski-Harabasz (Calinski & Harabasz, 1974)
  - Davies-Bouldin (Davies & Bouldin, 1979)
"""

import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)


# ═══════════════════════════════════════════════════════
# CENTROIDE (MacQueen 1967)
# "Center of mass of all points assigned to cluster"
# ═══════════════════════════════════════════════════════

class TestCentroid:

    def test_centroid_is_arithmetic_mean(self):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        centroid = points.mean(axis=0)
        np.testing.assert_allclose(centroid, [3.0, 4.0])

    def test_centroid_can_be_synthetic(self):
        """El centroide no necesariamente es un punto del dataset."""
        points = np.array([[0, 0], [4, 0], [0, 3]])
        centroid = points.mean(axis=0)  # [1.333, 1.0]
        is_in_dataset = any(np.allclose(centroid, p) for p in points)
        assert not is_in_dataset, "Centroide debería ser sintético aquí"

    def test_centroid_single_point(self):
        points = np.array([[5.0, 3.0]])
        centroid = points.mean(axis=0)
        np.testing.assert_allclose(centroid, [5.0, 3.0])

    def test_centroid_high_dimensional(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(100, 20))
        centroid = points.mean(axis=0)
        assert centroid.shape == (20,)
        # Centroide de Normal(0,1) debería estar cerca de 0
        assert np.all(np.abs(centroid) < 0.5)


# ═══════════════════════════════════════════════════════
# MEDOIDE (Kaufman & Rousseeuw 1990)
# "argmin_i Σ_j d(o_i, o_j)"
# ═══════════════════════════════════════════════════════

class TestMedoid:

    def _compute_medoid(self, points):
        dist_matrix = cdist(points, points)
        return dist_matrix.sum(axis=1).argmin()

    def test_medoid_minimizes_sum_distances(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(50, 3))
        medoid_idx = self._compute_medoid(points)
        sums = cdist(points, points).sum(axis=1)
        assert sums[medoid_idx] == sums.min()

    def test_medoid_is_real_point(self):
        points = np.array([[0, 0], [1, 0], [0.5, 0.1]])
        medoid_idx = self._compute_medoid(points)
        assert 0 <= medoid_idx < len(points)

    def test_medoid_symmetric_data(self):
        """En datos simétricos, medoide es el punto central."""
        points = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
        medoid_idx = self._compute_medoid(points)
        np.testing.assert_allclose(points[medoid_idx], [0, 0])

    def test_medoid_robust_to_outlier(self):
        points = np.array([[0, 0], [0.1, 0], [0, 0.1], [100, 100]])
        medoid_idx = self._compute_medoid(points)
        # Medoide no debería ser el outlier
        assert medoid_idx != 3

    def test_medoid_vs_centroid_with_outlier(self):
        """Medoide y centroide difieren significativamente con outliers."""
        points = np.array([[0, 0], [1, 0], [2, 0], [1000, 0]])
        medoid_idx = self._compute_medoid(points)
        centroid = points.mean(axis=0)
        # Centroide está en ~250, lejos de los puntos reales
        # Medoide debería estar en idx 1 o 2
        assert medoid_idx in [1, 2]
        assert centroid[0] > 200, "Centroide jalado por outlier"


# ═══════════════════════════════════════════════════════
# NEAREST-TO-CENTROID (Bien & Tibshirani 2011)
# "Real point closest to geometric centroid"
# ═══════════════════════════════════════════════════════

class TestNearestToCentroid:

    def _compute_ntc(self, points):
        centroid = points.mean(axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        return dists.argmin()

    def test_ntc_is_closest_to_centroid(self):
        points = np.array([[0, 0], [10, 0], [5, 0.1]])
        ntc_idx = self._compute_ntc(points)
        centroid = points.mean(axis=0)
        # [5, 0.1] es el más cercano al centroide [5, 0.033]
        assert ntc_idx == 2

    def test_ntc_is_real_point(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(100, 4))
        ntc_idx = self._compute_ntc(points)
        assert 0 <= ntc_idx < len(points)

    def test_ntc_equals_centroid_when_point_exists(self):
        """Si el centroide coincide con un punto, NTC lo devuelve."""
        points = np.array([[0, 0], [2, 0], [1, 0]])
        # Centroide = [1, 0] que es exactamente points[2]
        ntc_idx = self._compute_ntc(points)
        assert ntc_idx == 2

    def test_ntc_differs_from_medoid_asymmetric(self):
        """NTC ≠ Medoide en distribuciones asimétricas."""
        points = np.array([[0, 0], [1, 0], [2, 0], [100, 0]])
        ntc_idx = self._compute_ntc(points)
        dist_matrix = cdist(points, points)
        medoid_idx = dist_matrix.sum(axis=1).argmin()
        assert ntc_idx != medoid_idx, \
            f"NTC ({ntc_idx}) debería diferir de medoide ({medoid_idx})"

    def test_ntc_single_point(self):
        points = np.array([[42, 7]])
        ntc_idx = self._compute_ntc(points)
        assert ntc_idx == 0


# ═══════════════════════════════════════════════════════
# SILHOUETTE SCORE (Rousseeuw 1987)
# s(i) = (b(i) - a(i)) / max(a(i), b(i))
# ═══════════════════════════════════════════════════════

class TestSilhouette:

    def test_perfect_clusters_near_one(self):
        """Clusters perfectamente separados → Silhouette ≈ 1."""
        X = np.vstack([np.zeros((20, 2)), np.ones((20, 2)) * 100])
        labels = np.array([0]*20 + [1]*20)
        sil = silhouette_score(X, labels)
        assert sil > 0.99, f"Silhouette {sil:.4f} debería ser ≈1.0"

    def test_random_labels_near_zero(self):
        """Labels aleatorios → Silhouette cercano a 0."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 3))
        labels = rng.integers(0, 5, size=200)
        sil = silhouette_score(X, labels)
        assert -0.3 < sil < 0.3, f"Silhouette random {sil:.4f} debería estar ~0"

    def test_range_minus_one_to_one(self):
        """Silhouette ∈ [-1, 1]."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        labels = np.array([0]*50 + [1]*50)
        sil = silhouette_score(X, labels)
        assert -1 <= sil <= 1

    def test_needs_at_least_2_clusters(self):
        """Silhouette requiere ≥ 2 clusters."""
        X = np.random.randn(50, 2)
        with pytest.raises(ValueError):
            silhouette_score(X, np.zeros(50, dtype=int))


# ═══════════════════════════════════════════════════════
# CALINSKI-HARABASZ (1974)
# Higher is better
# ═══════════════════════════════════════════════════════

class TestCalinskiHarabasz:

    def test_higher_for_well_separated(self):
        """Clusters bien separados deben tener CH mayor."""
        rng = np.random.default_rng(42)
        # Bien separados
        X_good = np.vstack([rng.normal([0, 0], 0.1, (50, 2)),
                            rng.normal([10, 10], 0.1, (50, 2))])
        labels_good = np.array([0]*50 + [1]*50)
        # Mal separados
        X_bad = rng.normal(size=(100, 2))
        labels_bad = np.array([0]*50 + [1]*50)

        ch_good = calinski_harabasz_score(X_good, labels_good)
        ch_bad = calinski_harabasz_score(X_bad, labels_bad)
        assert ch_good > ch_bad

    def test_positive_value(self, scaled_data):
        X, _ = scaled_data
        labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        ch = calinski_harabasz_score(X, labels)
        assert ch > 0


# ═══════════════════════════════════════════════════════
# DAVIES-BOULDIN (1979)
# Lower is better
# ═══════════════════════════════════════════════════════

class TestDaviesBouldin:

    def test_lower_for_well_separated(self):
        """Clusters bien separados deben tener DB menor."""
        rng = np.random.default_rng(42)
        X_good = np.vstack([rng.normal([0, 0], 0.1, (50, 2)),
                            rng.normal([10, 10], 0.1, (50, 2))])
        labels_good = np.array([0]*50 + [1]*50)
        X_bad = rng.normal(size=(100, 2))
        labels_bad = np.array([0]*50 + [1]*50)

        db_good = davies_bouldin_score(X_good, labels_good)
        db_bad = davies_bouldin_score(X_bad, labels_bad)
        assert db_good < db_bad

    def test_non_negative(self, scaled_data):
        X, _ = scaled_data
        labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        db = davies_bouldin_score(X, labels)
        assert db >= 0


# ═══════════════════════════════════════════════════════
# METRICS INTEGRATION
# ═══════════════════════════════════════════════════════

class TestMetricsIntegration:
    """Verifica que las 3 métricas son coherentes entre sí."""

    def test_all_metrics_agree_on_quality(self, well_separated_clusters):
        X, true_labels = well_separated_clusters
        X_scaled = StandardScaler().fit_transform(X)

        km_good = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
        km_bad = KMeans(n_clusters=8, random_state=42).fit_predict(X_scaled)

        sil_good = silhouette_score(X_scaled, km_good)
        sil_bad = silhouette_score(X_scaled, km_bad)

        db_good = davies_bouldin_score(X_scaled, km_good)
        db_bad = davies_bouldin_score(X_scaled, km_bad)

        # Good clustering: higher silhouette, lower DB
        assert sil_good > sil_bad, "Silhouette debería ser mayor para k=3 que k=8"
        assert db_good < db_bad, "Davies-Bouldin debería ser menor para k=3 que k=8"
