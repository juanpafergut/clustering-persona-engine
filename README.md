# 🧬 Clustering Persona Engine v2

Motor de clustering y generación de avatares para segmentación de clientes, con chatbot IA integrado (Groq/Llama 3.3).

## Variables alineadas con BD v4

| Variable | Tipo | Fuente |
|----------|------|--------|
| `user_key` | INT PK | dim_user_behavior |
| `rfm_r_score` | FLOAT [0,1] | RFM normalizado |
| `rfm_f_score` | FLOAT [0,1] | RFM normalizado |
| `rfm_m_score` | FLOAT [0,1] | RFM normalizado |
| `rfm_composite` | FLOAT | Promedio RFM |
| `engagement_score` | FLOAT [1,10] | Lookalike similarity kernel |
| `total_spend_cents` | BIGINT | Monetario en centavos |
| `avg_session_duration_sec` | FLOAT | Duración sesión en segundos |
| `income_bracket` | VARCHAR | Rango de ingreso |

## Algoritmos implementados

- **K-Means++** (Arthur & Vassilvitskii, SODA 2007)
- **Mini-Batch K-Means** (Sculley, WWW 2010)
- **DBSCAN** (Ester et al., KDD 1996)
- **HDBSCAN** (Campello et al., PAKDD 2013)
- **Ward Hierarchical** (Murtagh & Legendre, 2014)
- **BIRCH** (Zhang et al., SIGMOD 1996)
- **K-Medoids/PAM** (Kaufman & Rousseeuw, 1990)

## Métodos de Avatar

- **Medoide** (PAM — Kaufman 1990)
- **Nearest-to-Centroid** (Bien & Tibshirani 2011)
- **Centroide** (MacQueen 1967)

## Setup local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Cloud

1. Push este repo a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repo → branch `main` → archivo `app.py`
4. Click **Deploy**

El chatbot pide la Groq API Key al usuario en la UI (no necesita secrets server-side).

## Chatbot IA (Groq + Llama 3.3 70B)

El tab "🤖 Chat IA" permite preguntar sobre los resultados del clustering.
Requiere una API Key gratuita de [console.groq.com](https://console.groq.com).

## Docker

```bash
docker build -t clustering-engine .
docker run -p 8501:8501 clustering-engine
```

---

📚 *SaleADS.ai — Clustering Persona Engine v2*
