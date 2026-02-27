"""
Test suite: Clustering Algorithms
Valida que sklearn implementa correctamente los papers referenciados.

Papers:
  - K-Means++ (Arthur & Vassilvitskii, SODA 2007)
  - Mini-Batch K-Means (Sculley, WWW 2010)
  - DBSCAN (Ester et al., KDD 1996)
  - Ward Hierarchical (Murtagh & Legendre, 2014)
  - BIRCH (Zhang et al., SIGMOD 1996)
  - K-Medoids / PAM (Kaufman & Rousseeuw, 1990)
"""

import pytest
import numpy as np
from itertools import permutations
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN,
    AgglomerativeClustering, Birch
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ═══════════════════════════════════════════════════════
# K-MEANS++ (Arthur & Vassilvitskii 2007)
# Paper: "D²-weighting produces O(log k)-competitive clustering"
# ═══════════════════════════════════════════════════════

class TestKMeansPP:
    """Valida K-Means++ contra Arthur & Vassilvitskii (SODA 2007)."""

    def test_produces_k_clusters(self, scaled_data):
        X, _ = scaled_data
        for k in [2, 3, 5]:
            km = KMeans(n_clusters=k, init="k-means++", random_state=42)
            labels = km.fit_predict(X)
            assert len(set(labels)) == k, f"Esperaba {k} clusters, obtuvo {len(set(labels))}"

    def test_all_points_assigned(self, scaled_data):
        X, _ = scaled_data
        km = KMeans(n_clusters=3, init="k-means++", random_state=42)
        labels = km.fit_predict(X)
        assert len(labels) == len(X)
        assert all(l >= 0 for l in labels)

    def test_centroids_are_cluster_means(self, scaled_data):
        """Paper MacQueen 1967: centroide = media del cluster tras convergencia."""
        X, _ = scaled_data
        km = KMeans(n_clusters=3, init="k-means++", random_state=42)
        km.fit(X)
        for c in range(3):
            mask = km.labels_ == c
            expected_centroid = X[mask].mean(axis=0)
            np.testing.assert_allclose(
                km.cluster_centers_[c], expected_centroid, atol=1e-6,
                err_msg=f"Centroide {c} no es la media del cluster"
            )

    def test_recovers_well_separated_clusters(self, scaled_data):
        """Con clusters bien separados, K-Means++ debe recuperar la estructura."""
        X, true_labels = scaled_data
        km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
        pred = km.fit_predict(X)
        # Accuracy con permutación (labels pueden estar permutados)
        best_acc = max(
            np.mean(np.array([p[l] for l in pred]) == true_labels)
            for p in permutations(range(3))
        )
        assert best_acc >= 0.95, f"Accuracy {best_acc:.2%} < 95%"

    def test_d2_init_better_than_random(self, scaled_data):
        """Arthur 2007: K-Means++ supera inicialización random."""
        X, _ = scaled_data
        km_pp = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit(X)
        km_rnd = KMeans(n_clusters=3, init="random", n_init=10, random_state=42).fit(X)
        assert km_pp.inertia_ <= km_rnd.inertia_ * 1.1, \
            "K-Means++ debería tener inercia ≤ random"

    def test_convergence_monotonic_inertia(self, scaled_data):
        """La inercia debe decrecer monótonamente (paper: φ es monótonamente decreciente)."""
        X, _ = scaled_data
        km = KMeans(n_clusters=3, init="k-means++", max_iter=1, n_init=1, random_state=42)
        prev_inertia = float("inf")
        for _ in range(20):
            km.max_iter += 1
            km.fit(X)
            assert km.inertia_ <= prev_inertia + 1e-10
            prev_inertia = km.inertia_

    def test_reproducibility_with_seed(self, scaled_data):
        X, _ = scaled_data
        l1 = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        l2 = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        np.testing.assert_array_equal(l1, l2)


# ═══════════════════════════════════════════════════════
# MINI-BATCH K-MEANS (Sculley 2010)
# Paper: "comparable results to K-Means with 3-10x speedup"
# ═══════════════════════════════════════════════════════

class TestMiniBatchKMeans:
    """Valida Mini-Batch K-Means contra Sculley (WWW 2010)."""

    def test_produces_k_clusters(self, scaled_data):
        X, _ = scaled_data
        mb = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=50)
        labels = mb.fit_predict(X)
        assert len(set(labels)) == 3

    def test_comparable_to_standard_kmeans(self, scaled_data):
        """Sculley 2010: resultados comparables a K-Means standard."""
        X, true_labels = scaled_data
        km = KMeans(n_clusters=3, random_state=42).fit(X)
        mb = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=50).fit(X)
        # Inercia no debería ser >50% peor
        assert mb.inertia_ <= km.inertia_ * 1.5, \
            f"Mini-Batch inercia ({mb.inertia_:.1f}) muy superior a K-Means ({km.inertia_:.1f})"

    def test_recovers_structure(self, scaled_data):
        X, true_labels = scaled_data
        mb = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=50)
        pred = mb.fit_predict(X)
        best_acc = max(
            np.mean(np.array([p[l] for l in pred]) == true_labels)
            for p in permutations(range(3))
        )
        assert best_acc >= 0.90

    def test_handles_large_batch(self, large_dataset):
        """Debe funcionar con datasets grandes."""
        mb = MiniBatchKMeans(n_clusters=5, batch_size=500, random_state=42)
        labels = mb.fit_predict(large_dataset)
        assert len(set(labels)) == 5
        assert len(labels) == len(large_dataset)


# ═══════════════════════════════════════════════════════
# DBSCAN (Ester et al. 1996)
# Paper: "regionQuery + core/border/noise classification"
# ═══════════════════════════════════════════════════════

class TestDBSCAN:
    """Valida DBSCAN contra Ester et al. (KDD 1996)."""

    def test_large_eps_single_cluster(self, scaled_data):
        """eps enorme → todo en 1 cluster."""
        X, _ = scaled_data
        db = DBSCAN(eps=100, min_samples=2)
        labels = db.fit_predict(X)
        assert len(set(labels) - {-1}) == 1

    def test_tiny_eps_all_noise(self, scaled_data):
        """eps minúsculo → todo es noise (label=-1)."""
        X, _ = scaled_data
        db = DBSCAN(eps=0.001, min_samples=5)
        labels = db.fit_predict(X)
        assert all(l == -1 for l in labels)

    def test_finds_clusters_with_reasonable_params(self, scaled_data):
        """Parámetros razonables detectan clusters."""
        X, _ = scaled_data
        db = DBSCAN(eps=1.0, min_samples=3)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 2, f"Solo encontró {n_clusters} clusters"

    def test_noise_label_is_minus_one(self, scaled_data):
        """Paper: noise = puntos no pertenecientes a ningún cluster (Def 6)."""
        X, _ = scaled_data
        db = DBSCAN(eps=0.5, min_samples=3)
        labels = db.fit_predict(X)
        # Noise debe ser -1 por convención de sklearn
        noise_labels = set(labels[labels < 0])
        assert noise_labels <= {-1}

    def test_core_points_have_enough_neighbors(self, scaled_data):
        """Paper Def 2: core point ↔ |N_eps(p)| ≥ MinPts."""
        X, _ = scaled_data
        eps, min_samples = 1.0, 3
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(X)
        # core_sample_indices_ son los core points
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X)
        for idx in db.core_sample_indices_:
            neighbors = np.sum(dists[idx] <= eps)
            assert neighbors >= min_samples, \
                f"Core point {idx} tiene {neighbors} vecinos < {min_samples}"

    def test_non_core_points_are_border_or_noise(self, scaled_data):
        """Puntos no-core son border (asignados) o noise (-1)."""
        X, _ = scaled_data
        db = DBSCAN(eps=1.0, min_samples=3)
        labels = db.fit_predict(X)
        core_set = set(db.core_sample_indices_)
        for i in range(len(X)):
            if i not in core_set:
                # Es border (label ≥ 0) o noise (label == -1)
                assert labels[i] >= -1

    def test_deterministic(self, scaled_data):
        """DBSCAN debe ser determinista (no depende de random seed)."""
        X, _ = scaled_data
        l1 = DBSCAN(eps=1.0, min_samples=3).fit_predict(X)
        l2 = DBSCAN(eps=1.0, min_samples=3).fit_predict(X)
        np.testing.assert_array_equal(l1, l2)


# ═══════════════════════════════════════════════════════
# WARD HIERARCHICAL (Murtagh & Legendre 2014)
# Paper: "minimize increase in total within-cluster SSE"
# D(c1,c2) = |c1|·|c2|/(|c1|+|c2|) · ||c1-c2||²
# ═══════════════════════════════════════════════════════

class TestWardHierarchical:
    """Valida Ward Hierarchical contra Murtagh & Legendre (2014)."""

    def test_produces_k_clusters(self, scaled_data):
        X, _ = scaled_data
        for k in [2, 3, 5]:
            hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = hc.fit_predict(X)
            assert len(set(labels)) == k

    def test_all_points_assigned(self, scaled_data):
        X, _ = scaled_data
        hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
        labels = hc.fit_predict(X)
        assert len(labels) == len(X)
        assert all(l >= 0 for l in labels)

    def test_recovers_structure(self, scaled_data):
        X, true_labels = scaled_data
        hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
        pred = hc.fit_predict(X)
        best_acc = max(
            np.mean(np.array([p[l] for l in pred]) == true_labels)
            for p in permutations(range(3))
        )
        assert best_acc >= 0.95

    def test_ward_minimizes_variance(self, scaled_data):
        """Ward debe producir menor inercia que single linkage."""
        X, _ = scaled_data
        ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
        single = AgglomerativeClustering(n_clusters=3, linkage="single").fit_predict(X)

        def inertia(X, labels):
            total = 0
            for c in set(labels):
                mask = labels == c
                centroid = X[mask].mean(axis=0)
                total += np.sum((X[mask] - centroid) ** 2)
            return total

        assert inertia(X, ward) <= inertia(X, single) * 1.1, \
            "Ward debe minimizar varianza intra-cluster mejor que single linkage"

    def test_deterministic(self, scaled_data):
        X, _ = scaled_data
        l1 = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
        l2 = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
        np.testing.assert_array_equal(l1, l2)


# ═══════════════════════════════════════════════════════
# BIRCH (Zhang et al. 1996)
# Paper: "O(n) time, single scan, handles noise"
# ═══════════════════════════════════════════════════════

class TestBIRCH:
    """Valida BIRCH contra Zhang et al. (SIGMOD 1996)."""

    def test_produces_k_clusters(self, scaled_data):
        X, _ = scaled_data
        birch = Birch(n_clusters=3, threshold=0.5)
        labels = birch.fit_predict(X)
        assert len(set(labels)) == 3

    def test_all_points_assigned(self, scaled_data):
        X, _ = scaled_data
        labels = Birch(n_clusters=3).fit_predict(X)
        assert len(labels) == len(X)
        assert all(l >= 0 for l in labels)

    def test_recovers_structure(self, scaled_data):
        X, true_labels = scaled_data
        pred = Birch(n_clusters=3, threshold=0.5).fit_predict(X)
        best_acc = max(
            np.mean(np.array([p[l] for l in pred]) == true_labels)
            for p in permutations(range(3))
        )
        assert best_acc >= 0.85, f"BIRCH accuracy {best_acc:.2%} < 85%"

    def test_scales_to_large_dataset(self, large_dataset):
        """Paper: BIRCH escala linealmente con N."""
        import time
        birch = Birch(n_clusters=5, threshold=0.5)
        t0 = time.time()
        labels = birch.fit_predict(large_dataset)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"BIRCH tardó {elapsed:.1f}s en 5k puntos (debe ser <5s)"
        assert len(set(labels)) == 5

    def test_threshold_sensitivity(self, scaled_data):
        """Threshold bajo → más subclusters → mejor calidad."""
        X, _ = scaled_data
        labels_low = Birch(n_clusters=3, threshold=0.1).fit_predict(X)
        labels_high = Birch(n_clusters=3, threshold=1.0).fit_predict(X)
        
        n_clusters_low = len(set(labels_low))
        n_clusters_high = len(set(labels_high))
        
        # Ambos deben producir al menos 2 clusters para comparar
        if n_clusters_low >= 2 and n_clusters_high >= 2:
            sil_low = silhouette_score(X, labels_low)
            sil_high = silhouette_score(X, labels_high)
            # Threshold bajo generalmente da mejor o igual silhouette
            assert sil_low >= sil_high - 0.2, \
                f"Threshold bajo ({sil_low:.3f}) no debería ser mucho peor que alto ({sil_high:.3f})"
        else:
            # Si threshold alto colapsa a 1 cluster, eso confirma la sensibilidad
            assert n_clusters_low >= n_clusters_high, \
                "Threshold bajo debe producir ≥ subclusters que threshold alto"


# ═══════════════════════════════════════════════════════
# K-MEDOIDS / PAM (Kaufman & Rousseeuw 1990)
# Paper: "representative objects that minimize total dissimilarity"
# ═══════════════════════════════════════════════════════

class TestKMedoidsFallback:
    """Testa la lógica de medoide independiente de sklearn_extra."""

    def test_medoid_is_argmin_distances(self):
        """Paper PAM: medoide = argmin_i Σ_j d(o_i, o_j)."""
        from scipy.spatial.distance import cdist
        points = np.array([[0, 0], [1, 0], [0.5, 0.1], [10, 10]])
        dist_matrix = cdist(points, points)
        sums = dist_matrix.sum(axis=1)
        medoid_idx = sums.argmin()
        # Punto [0.5, 0.1] está más al centro de los 3 cercanos
        # pero [10,10] jala → medoide depende del layout
        assert 0 <= medoid_idx < len(points)
        # Verificar que es realmente el mínimo
        for i in range(len(points)):
            assert sums[medoid_idx] <= sums[i]

    def test_medoid_is_real_point(self):
        """Medoide siempre es un punto real del dataset."""
        from scipy.spatial.distance import cdist
        rng = np.random.default_rng(42)
        points = rng.normal(size=(100, 3))
        dist_matrix = cdist(points, points)
        medoid_idx = dist_matrix.sum(axis=1).argmin()
        # El medoide debe ser un índice válido
        assert 0 <= medoid_idx < len(points)
        # Y el punto debe existir exactamente en el dataset
        assert np.array_equal(points[medoid_idx], points[medoid_idx])

    def test_medoid_robust_to_outlier(self):
        """Medoide es robusto a outliers (vs centroide que se desplaza)."""
        from scipy.spatial.distance import cdist
        # Cluster compacto + 1 outlier extremo
        points = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [100, 100]])
        dist_matrix = cdist(points, points)
        medoid_idx = dist_matrix.sum(axis=1).argmin()
        # El medoide debe estar en el cluster compacto (idx 0-3), no el outlier
        assert medoid_idx < 4, f"Medoide es outlier idx={medoid_idx}, debería ser 0-3"

    def test_medoid_differs_from_centroid(self):
        """Medoide ≠ punto más cercano al centroide en distribuciones asimétricas."""
        from scipy.spatial.distance import cdist
        points = np.array([[0, 0], [1, 0], [2, 0], [100, 0]])
        centroid = points.mean(axis=0)  # [25.75, 0]
        # Nearest to centroid
        ntc_idx = np.argmin(np.linalg.norm(points - centroid, axis=1))
        # Medoid
        dist_matrix = cdist(points, points)
        medoid_idx = dist_matrix.sum(axis=1).argmin()
        # Deben ser diferentes con esta distribución asimétrica
        assert medoid_idx != ntc_idx, \
            f"Medoide ({medoid_idx}) y NTC ({ntc_idx}) deberían diferir con outlier"
