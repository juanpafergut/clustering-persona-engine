"""
Clustering Persona Engine — Production App
Deploy: Streamlit Cloud / Docker / Local

Papers:
  K-Means++ (Arthur & Vassilvitskii, SODA 2007)
  DBSCAN (Ester et al., KDD 1996)
  HDBSCAN (Campello et al., PAKDD 2013)
  Ward Hierarchical (Murtagh & Legendre, 2014)
  Mini-Batch K-Means (Sculley, WWW 2010)
  k-Medoids/PAM (Kaufman & Rousseeuw, 1990)
  Silhouette (Rousseeuw, 1987)
  BIRCH (Zhang et al., SIGMOD 1996)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# ─── CONFIG ───
st.set_page_config(
    page_title="Clustering Persona Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #0f1117);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 28px; font-weight: 700; }
    .metric-label { font-size: 11px; color: #94a3b8; margin-top: 4px; }
    .avatar-card {
        background: linear-gradient(135deg, #1a1d2e, #0f1117);
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)


# ─── FAKE DATA GENERATOR ───
@st.cache_data
def generate_fake_data(n_samples=500, seed=42):
    """Generate fake data aligned with dim_user_behavior schema (Variables_Cruce_BD_Leo_v1.md §2.1)."""
    rng = np.random.default_rng(seed)
    
    archetypes = [
        {"name": "Premium", "r": 0.9, "f": 30, "m": 60000, "age": 32, "eng": 8.5, "session": 1500},
        {"name": "Frequent", "r": 0.7, "f": 18, "m": 28000, "age": 28, "eng": 7.2, "session": 1080},
        {"name": "At-Risk", "r": 0.2, "f": 3, "m": 8000, "age": 48, "eng": 4.5, "session": 300},
        {"name": "New", "r": 0.8, "f": 2, "m": 4500, "age": 25, "eng": 6.8, "session": 720},
        {"name": "Dormant", "r": 0.05, "f": 1, "m": 1500, "age": 52, "eng": 3.2, "session": 120},
    ]
    
    names_m = ["Carlos","Andrés","Miguel","Juan","Diego","Luis","Pedro","Tomás","Rafael","Santiago"]
    names_f = ["María","Sofía","Lucía","Valentina","Camila","Laura","Ana","Isabella","Natalia","Gabriela"]
    cities = ["Bogotá","Medellín","CDMX","Lima","Santiago","Buenos Aires","São Paulo","Quito"]
    countries = ["CO","CO","MX","PE","CL","AR","BR","EC"]
    channels = ["Email","Social","Organic","Paid","Referral","Direct"]
    devices = ["Mobile","Desktop","Tablet"]
    income_brackets = ["<30K","30K-60K","60K-100K","100K+"]
    
    data = []
    for i in range(n_samples):
        arch = archetypes[i % len(archetypes)]
        s = 0.25
        is_f = rng.random() > 0.5
        name = rng.choice(names_f if is_f else names_m)
        city_idx = rng.integers(0, len(cities))
        
        # RFM scores normalized [0,1] per .md §2.1: rfm_r_score, rfm_f_score, rfm_m_score
        rfm_r = round(np.clip(arch["r"] + rng.normal(0, 0.15), 0, 1), 3)
        rfm_f = round(np.clip(arch["f"] / 40 + rng.normal(0, 0.1), 0, 1), 3)
        rfm_m = round(np.clip(arch["m"] / 80000 + rng.normal(0, 0.1), 0, 1), 3)
        
        data.append({
            "user_key": i + 1,
            "business_key": rng.integers(1, 101),
            "avatar_name": f"{name} {chr(65 + rng.integers(0, 26))}.",
            "age": int(np.clip(arch["age"] + rng.normal(0, 8), 18, 75)),
            "gender": "F" if is_f else "M",
            "country_code": countries[city_idx],
            "city": cities[city_idx],
            "acquisition_channel": rng.choice(channels),
            "device_type": rng.choice(devices),
            "income_bracket": rng.choice(income_brackets),
            "total_interactions": int(np.clip(arch["f"] * (1 + rng.normal(0, s)), 1, 80)),
            "total_spend_cents": int(np.clip(arch["m"] * (1 + rng.normal(0, s)), 500, 200000)),
            "avg_session_duration_sec": round(np.clip(arch["session"] * (1 + rng.normal(0, 0.3)), 30, 3600), 1),
            "engagement_score": round(np.clip(arch["eng"] + rng.normal(0, 1), 1, 10), 1),
            "rfm_r_score": rfm_r,
            "rfm_f_score": rfm_f,
            "rfm_m_score": rfm_m,
            "rfm_composite": round((rfm_r + rfm_f + rfm_m) / 3, 3),
        })
    
    return pd.DataFrame(data)


# ─── CLUSTERING ENGINE ───
def run_clustering(df, features, method, params):
    """Run clustering with scikit-learn (production-grade)."""
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "K-Means++":
        model = KMeans(
            n_clusters=params["k"], init="k-means++", n_init=10,
            max_iter=300, random_state=42
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "Mini-Batch K-Means":
        model = MiniBatchKMeans(
            n_clusters=params["k"], init="k-means++",
            batch_size=min(1000, len(X)), random_state=42
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_pts"])
        labels = model.fit_predict(X_scaled)
        
    elif method == "HDBSCAN" and HAS_HDBSCAN:
        model = hdbscan.HDBSCAN(
            min_cluster_size=params.get("min_cluster_size", 15),
            min_samples=params.get("min_samples", 5)
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "Ward Hierarchical":
        model = AgglomerativeClustering(
            n_clusters=params["k"], linkage="ward"
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "BIRCH":
        model = Birch(
            n_clusters=params["k"],
            threshold=params.get("threshold", 0.5),
            branching_factor=params.get("branching_factor", 50)
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "K-Medoids (PAM)" and HAS_KMEDOIDS:
        model = KMedoids(
            n_clusters=params["k"], method="pam",
            init="k-medoids++", random_state=42
        )
        labels = model.fit_predict(X_scaled)
        
    elif method == "K-Medoids (PAM)" and not HAS_KMEDOIDS:
        # Fallback: K-Means with medoid post-processing
        model = KMeans(n_clusters=params["k"], init="k-means++", random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        labels = KMeans(n_clusters=params.get("k", 4), random_state=42).fit_predict(X_scaled)
    
    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    
    return labels, X_scaled, X_2d, pca


def compute_metrics(X_scaled, labels):
    """Compute clustering quality metrics."""
    valid = labels >= 0
    n_clusters = len(set(labels[valid]))
    n_noise = np.sum(labels < 0)
    
    metrics = {"n_clusters": n_clusters, "n_noise": n_noise}
    
    if n_clusters >= 2 and valid.sum() > n_clusters:
        metrics["silhouette"] = silhouette_score(X_scaled[valid], labels[valid])
        metrics["calinski_harabasz"] = calinski_harabasz_score(X_scaled[valid], labels[valid])
        metrics["davies_bouldin"] = davies_bouldin_score(X_scaled[valid], labels[valid])
    else:
        metrics["silhouette"] = 0
        metrics["calinski_harabasz"] = 0
        metrics["davies_bouldin"] = float("inf")
    
    return metrics


def compute_avatars(df, labels, features, method="medoid"):
    """Compute cluster representatives."""
    avatars = []
    unique_labels = sorted(set(labels[labels >= 0]))
    
    for cl in unique_labels:
        mask = labels == cl
        cluster_df = df[mask]
        cluster_vals = cluster_df[features].values
        
        if method == "medoid":
            # PAM: argmin_i Σ_j d(i,j) — Kaufman & Rousseeuw 1990
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(cluster_vals, cluster_vals)
            sums = dist_matrix.sum(axis=1)
            rep_idx = sums.argmin()
        elif method == "nearest_centroid":
            # Nearest-to-Centroid — Bien & Tibshirani 2011
            centroid = cluster_vals.mean(axis=0)
            dists = np.linalg.norm(cluster_vals - centroid, axis=1)
            rep_idx = dists.argmin()
        else:  # centroid → pick nearest real point
            centroid = cluster_vals.mean(axis=0)
            dists = np.linalg.norm(cluster_vals - centroid, axis=1)
            rep_idx = dists.argmin()
        
        rep = cluster_df.iloc[rep_idx]
        
        # Cluster stats
        stats = {}
        for f in features:
            stats[f] = {"mean": cluster_df[f].mean(), "std": cluster_df[f].std()}
        
        avatars.append({
            "cluster": cl,
            "size": mask.sum(),
            "pct": round(mask.sum() / len(labels) * 100, 1),
            "representative": rep.to_dict(),
            "stats": stats,
        })
    
    return avatars


# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### 🧬 Clustering Engine")
    st.markdown("---")
    
    # Data source
    data_source = st.radio("📁 Datos", ["Generar fake data", "Subir CSV"], index=0)
    
    if data_source == "Generar fake data":
        n_samples = st.slider("Muestras", 100, 5000, 500, 100)
        df = generate_fake_data(n_samples)
        numeric_cols = ["rfm_r_score", "rfm_f_score", "rfm_m_score", "rfm_composite",
                        "age", "avg_session_duration_sec", "engagement_score",
                        "total_interactions", "total_spend_cents"]
    else:
        uploaded = st.file_uploader("CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            st.info("Sube un CSV para comenzar")
            st.stop()
    
    st.markdown("---")
    
    # Features
    features = st.multiselect("📊 Variables", numeric_cols, 
                              default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
    
    if len(features) < 2:
        st.warning("Selecciona al menos 2 variables")
        st.stop()
    
    st.markdown("---")
    
    # Clustering method
    methods_available = ["K-Means++", "Mini-Batch K-Means", "DBSCAN", 
                         "Ward Hierarchical", "BIRCH", "K-Medoids (PAM)"]
    if HAS_HDBSCAN:
        methods_available.insert(3, "HDBSCAN")
    
    method = st.selectbox("🔬 Método clustering", methods_available)
    
    params = {}
    if method in ["K-Means++", "Mini-Batch K-Means", "Ward Hierarchical", "BIRCH", "K-Medoids (PAM)"]:
        params["k"] = st.slider("K clusters", 2, 10, 4)
    if method == "DBSCAN":
        params["eps"] = st.slider("ε (eps)", 0.1, 3.0, 0.5, 0.1)
        params["min_pts"] = st.slider("min_samples", 2, 20, 5)
    if method == "HDBSCAN" and HAS_HDBSCAN:
        params["min_cluster_size"] = st.slider("min_cluster_size", 5, 50, 15)
        params["min_samples"] = st.slider("min_samples", 2, 20, 5)
    if method == "BIRCH":
        params["threshold"] = st.slider("threshold", 0.1, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    
    avatar_method = st.selectbox("🧑‍💼 Avatar", 
                                  ["medoid", "nearest_centroid", "centroid"],
                                  format_func=lambda x: {
                                      "medoid": "Medoide (PAM — Kaufman 1990)",
                                      "nearest_centroid": "Nearest-to-Centroid (Bien 2011)",
                                      "centroid": "Centroide (MacQueen 1967)"
                                  }[x])
    
    st.markdown("---")
    st.markdown(f"📐 **{len(df):,}** registros × **{len(features)}** variables")


# ─── RUN CLUSTERING ───
labels, X_scaled, X_2d, pca = run_clustering(df, features, method, params)
metrics = compute_metrics(X_scaled, labels)
avatars = compute_avatars(df, labels, features, avatar_method)

df["cluster_id"] = labels
df["cluster_method"] = method

# Compute per-point silhouette if possible
try:
    from sklearn.metrics import silhouette_samples
    valid = labels >= 0
    if metrics["n_clusters"] >= 2 and valid.sum() > metrics["n_clusters"]:
        sil_samples = np.full(len(labels), np.nan)
        sil_samples[valid] = silhouette_samples(X_scaled[valid], labels[valid])
        df["silhouette_score"] = sil_samples
    else:
        df["silhouette_score"] = np.nan
except:
    df["silhouette_score"] = np.nan

# Centroid distance per point
from scipy.spatial.distance import cdist as _cdist
if method not in ["DBSCAN", "HDBSCAN"]:
    centroids = np.array([X_scaled[labels == c].mean(axis=0) for c in sorted(set(labels[labels >= 0]))])
    dists_to_centroid = np.full(len(labels), np.nan)
    for c in sorted(set(labels[labels >= 0])):
        mask = labels == c
        dists_to_centroid[mask] = np.linalg.norm(X_scaled[mask] - centroids[c], axis=1)
    df["centroid_distance"] = np.round(dists_to_centroid, 4)
else:
    df["centroid_distance"] = np.nan

# ─── HEADER ───
st.markdown("# 🧬 Clustering Persona Engine")
st.markdown(f"**{method}** · {len(features)} variables · {len(df):,} registros · "
            f"[Papers: Arthur 2007, Ester 1996, Murtagh 2014, Kaufman 1990]")

# ─── METRICS ROW ───
cols = st.columns(4)
with cols[0]:
    st.metric("Clusters", metrics["n_clusters"])
with cols[1]:
    st.metric("Silhouette", f"{metrics['silhouette']:.3f}")
with cols[2]:
    st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.0f}")
with cols[3]:
    db_val = metrics["davies_bouldin"]
    st.metric("Davies-Bouldin", f"{db_val:.3f}" if db_val < 100 else "N/A")

# ─── TABS ───
tab_scatter, tab_avatars, tab_elbow, tab_data, tab_chat, tab_export = st.tabs(
    ["📊 Scatter Plot", "🧑‍💼 Avatars", "📈 Optimal K", "📋 Datos", "🤖 Chat IA", "💾 Exportar"]
)

# ─── SCATTER ───
with tab_scatter:
    df_plot = pd.DataFrame({
        "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
        "Cluster": labels.astype(str),
    })
    # Add avatar info
    for col in ["avatar_name", "city", "acquisition_channel"]:
        if col in df.columns:
            df_plot[col] = df[col].values
    
    fig = px.scatter(
        df_plot[df_plot["Cluster"] != "-1"], x="PC1", y="PC2", color="Cluster",
        hover_data=[c for c in ["avatar_name", "city", "acquisition_channel"] if c in df_plot.columns],
        title=f"PCA 2D — {method} (Silhouette: {metrics['silhouette']:.3f})",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark",
        height=500,
    )
    
    # Noise points
    noise_mask = df_plot["Cluster"] == "-1"
    if noise_mask.any():
        fig.add_trace(go.Scatter(
            x=df_plot.loc[noise_mask, "PC1"], y=df_plot.loc[noise_mask, "PC2"],
            mode="markers", marker=dict(color="gray", size=4, opacity=0.4),
            name="Noise", showlegend=True
        ))
    
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    
    var_explained = pca.explained_variance_ratio_
    st.caption(f"PCA: PC1 = {var_explained[0]:.1%}, PC2 = {var_explained[1]:.1%} varianza explicada")

# ─── AVATARS ───
with tab_avatars:
    avatar_labels = {"medoid": "Medoide (PAM)", "nearest_centroid": "Nearest-to-Centroid", "centroid": "Centroide"}
    st.markdown(f"**Método representante:** {avatar_labels[avatar_method]}")
    
    for av in avatars:
        rep = av["representative"]
        color = px.colors.qualitative.Set2[av["cluster"] % len(px.colors.qualitative.Set2)]
        
        with st.expander(f"🎯 Cluster {av['cluster']} — {av['size']} clientes ({av['pct']}%)", expanded=True):
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown(f"**Avatar:** {rep.get('avatar_name', 'N/A')}")
                st.markdown(f"**Edad:** {rep.get('age', 'N/A')}")
                st.markdown(f"**Ciudad:** {rep.get('city', 'N/A')}")
                st.markdown(f"**Canal:** {rep.get('acquisition_channel', 'N/A')}")
                st.markdown(f"**Device:** {rep.get('device_type', 'N/A')}")
                st.markdown(f"**Income:** {rep.get('income_bracket', 'N/A')}")
            
            with c2:
                stats_df = pd.DataFrame({
                    "Variable": features,
                    "Media": [av["stats"][f]["mean"] for f in features],
                    "Std": [av["stats"][f]["std"] for f in features],
                }).round(2)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

# ─── OPTIMAL K (Elbow + Silhouette) ───
with tab_elbow:
    st.markdown("### Búsqueda de K óptimo")
    st.markdown("Combina Elbow Method (inercia) + Silhouette Score para encontrar el mejor K.")
    
    if method not in ["DBSCAN", "HDBSCAN"]:
        k_range = range(2, 11)
        inertias, sils = [], []
        
        X_elbow = StandardScaler().fit_transform(df[features].values)
        
        for k in k_range:
            km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
            km.fit(X_elbow)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X_elbow, km.labels_))
        
        col_e, col_s = st.columns(2)
        
        with col_e:
            fig_e = px.line(x=list(k_range), y=inertias, markers=True,
                           title="Elbow Method (Inercia)", template="plotly_dark",
                           labels={"x": "K", "y": "Inercia"})
            st.plotly_chart(fig_e, use_container_width=True)
        
        with col_s:
            fig_s = px.bar(x=list(k_range), y=sils,
                          title="Silhouette Score por K", template="plotly_dark",
                          labels={"x": "K", "y": "Silhouette"})
            best_k = list(k_range)[np.argmax(sils)]
            st.plotly_chart(fig_s, use_container_width=True)
        
        st.success(f"**K óptimo sugerido: {best_k}** (Silhouette = {max(sils):.3f})")
    else:
        st.info("Elbow method aplica a métodos con K fijo. DBSCAN/HDBSCAN determinan K automáticamente.")

# ─── DATA ───
with tab_data:
    st.dataframe(df.head(100), use_container_width=True, height=400)
    st.caption(f"Mostrando 100 de {len(df):,} registros")

# ─── CHATBOT IA ───
with tab_chat:
    st.markdown("### 🤖 Asistente de Clustering")
    st.markdown("Pregunta sobre los resultados, los clusters, los avatares, o los métodos usados.")
    
    # API Key config
    api_key = st.text_input("🔑 Groq API Key", type="password", 
                            help="Obtén tu key en console.groq.com")
    
    if not api_key:
        st.info("Ingresa tu API Key de Groq para activar el chatbot.")
        st.markdown("""
        **¿Cómo obtener una API Key?**
        1. Ve a [console.groq.com](https://console.groq.com)
        2. Crea una cuenta gratuita
        3. Genera una API Key en API Keys
        """)
    else:
        from groq import Groq
        
        # Build context about current clustering results
        cluster_summary = []
        for av in avatars:
            rep = av["representative"]
            cluster_summary.append(
                f"Cluster {av['cluster']}: {av['size']} usuarios ({av['pct']}%), "
                f"avatar={rep.get('avatar_name','N/A')}, ciudad={rep.get('city','N/A')}, "
                f"income={rep.get('income_bracket','N/A')}, "
                + ", ".join(f"{f}={av['stats'][f]['mean']:.2f}±{av['stats'][f]['std']:.2f}" for f in features)
            )
        
        system_ctx = f"""Eres un analista de clustering experto de SaleADS.ai. 
Responde en español, de forma concisa y técnica.

CONTEXTO DEL ANÁLISIS ACTUAL:
- Método: {method}
- Variables: {', '.join(features)}
- Registros: {len(df):,}
- Métricas: Silhouette={metrics['silhouette']:.3f}, CH={metrics['calinski_harabasz']:.0f}, DB={metrics['davies_bouldin']:.3f}
- Clusters encontrados: {metrics['n_clusters']}
- Noise points: {metrics['n_noise']}

RESUMEN DE CLUSTERS:
{chr(10).join(cluster_summary)}

PAPERS DE REFERENCIA:
K-Means++ (Arthur & Vassilvitskii, SODA 2007), DBSCAN (Ester et al., KDD 1996),
HDBSCAN (Campello et al., PAKDD 2013), Ward (Murtagh & Legendre, 2014),
BIRCH (Zhang et al., SIGMOD 1996), PAM (Kaufman & Rousseeuw, 1990),
Silhouette (Rousseeuw, 1987), Data-Driven Personas (An et al., ACM TWEB 2018).

Variables alineadas con dim_user_behavior del esquema v4 de SaleADS."""

        # Chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Pregunta sobre los clusters..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analizando..."):
                    try:
                        client = Groq(api_key=api_key)
                        messages = [{"role": "system", "content": system_ctx}]
                        messages += [{"role": m["role"], "content": m["content"]} 
                                    for m in st.session_state.chat_messages]
                        
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages,
                            max_tokens=1024,
                            temperature=0.3,
                        )
                        
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        err = str(e)
                        if "auth" in err.lower() or "401" in err:
                            st.error("❌ API Key inválida. Verifica en console.groq.com")
                        elif "rate" in err.lower() or "429" in err:
                            st.error("⏳ Rate limit alcanzado. Groq free tier tiene límites por minuto.")
                        else:
                            st.error(f"Error: {err}")
        
        if st.session_state.chat_messages:
            if st.button("🗑️ Limpiar chat"):
                st.session_state.chat_messages = []
                st.rerun()

# ─── EXPORT ───
with tab_export:
    st.markdown("### 💾 Exportar resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Descargar datos + clusters (CSV)",
            csv_data, "clustering_results.csv", "text/csv"
        )
    
    with col2:
        avatar_df = pd.DataFrame([
            {
                "cluster": av["cluster"],
                "size": av["size"],
                "pct": av["pct"],
                "avatar_name": av["representative"].get("avatar_name", ""),
                "avatar_age": av["representative"].get("age", ""),
                "avatar_city": av["representative"].get("city", ""),
                "income_bracket": av["representative"].get("income_bracket", ""),
                **{f"mean_{f}": av["stats"][f]["mean"] for f in features},
            }
            for av in avatars
        ])
        csv_avatars = avatar_df.to_csv(index=False)
        st.download_button(
            "📥 Descargar avatars (CSV)",
            csv_avatars, "avatars.csv", "text/csv"
        )

# ─── FOOTER ───
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 11px; color: #64748b;'>
📚 K-Means++ (Arthur 2007) · DBSCAN (Ester 1996) · HDBSCAN (Campello 2013) · 
Ward (Murtagh 2014) · BIRCH (Zhang 1996) · PAM (Kaufman 1990) · 
Mini-Batch K-Means (Sculley 2010) · Silhouette (Rousseeuw 1987)
</div>
""", unsafe_allow_html=True)
