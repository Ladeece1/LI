
import re
import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import community as community_louvain
except Exception:
    community_louvain = None

st.set_page_config(page_title="LinkedIn Network Explorer", layout="wide")

# ---------- Utilities ----------
def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def guess_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def find_one(cands):
        for c in cands:
            if c.lower() in cols: return cols[c.lower()]
        for key in cols:
            for c in cands:
                if c.lower() in key: return cols[key]
        return None
    return dict(
        first_name=find_one(["first name","firstname","given name","s","first"]),
        last_name=find_one(["last name","lastname","surname","family name"]),
        url=find_one(["url","profile url","linkedin url"]),
        email=find_one(["email address","email"]),
        company=find_one(["company","organization","employer"]),
        position=find_one(["position","title","role","job title","headline"]),
        connected_on=find_one(["connected on","connected","date","connected_on"]),
    )

def name_from_row(row, cols):
    fn = clean_text(row.get(cols["first_name"])) if cols["first_name"] else ""
    ln = clean_text(row.get(cols["last_name"])) if cols["last_name"] else ""
    if fn or ln: return f"{fn} {ln}".strip()
    url = clean_text(row.get(cols["url"])) if cols["url"] else ""
    if url:
        slug = url.rstrip("/").split("/")[-1].replace("-"," ").replace("_"," ")
        if slug and len(slug)>1: return slug.title()
    email = clean_text(row.get(cols["email"])) if cols["email"] else ""
    if "@" in email: return email.split("@")[0]
    return "Unknown"

def domain_from_email(email):
    if pd.isna(email): return ""
    return str(email).split("@")[-1].lower() if "@" in str(email) else ""

def build_graph(df, cols):
    # Fixed, simple inference rules (no UI):
    USE_COMPANY = True
    USE_DOMAIN = True
    USE_POSITION = True
    POSITION_THRESHOLD = 0.60
    W_COMPANY = 3.0
    W_DOMAIN = 2.0
    W_POSITION = 1.0

    G = nx.Graph()
    YOU = "__YOU__"
    G.add_node(YOU, label="You", group="You", size=22)

    rows = []
    for i, r in df.iterrows():
        name = name_from_row(r, cols)
        url = clean_text(r.get(cols["url"])) if cols["url"] else ""
        email = clean_text(r.get(cols["email"])) if cols["email"] else ""
        company = clean_text(r.get(cols["company"])) if cols["company"] else ""
        position = clean_text(r.get(cols["position"])) if cols["position"] else ""
        node_id = f"n{i}"
        rows.append({
            "id": node_id, "name": name, "url": url, "email": email,
            "domain": domain_from_email(email), "company": company, "position": position
        })
        G.add_node(node_id, label=name, title=f"{name}<br>{company} — {position}",
                   url=url, company=company, position=position, domain=domain_from_email(email), size=8)
        G.add_edge(YOU, node_id, weight=1.0, kind="ego")

    df_nodes = pd.DataFrame(rows)

    # Position similarity
    sims = None
    if USE_POSITION and len(df_nodes) > 1:
        texts = df_nodes["position"].fillna("").astype(str).tolist()
        if any(t.strip() for t in texts):
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            X = vec.fit_transform(texts)
            sims = cosine_similarity(X)

    n = len(df_nodes)
    for i in range(n):
        for j in range(i+1, n):
            w = 0.0
            if USE_COMPANY:
                ci = df_nodes.loc[i,"company"].strip().lower() if isinstance(df_nodes.loc[i,"company"], str) else ""
                cj = df_nodes.loc[j,"company"].strip().lower() if isinstance(df_nodes.loc[j,"company"], str) else ""
                if ci and ci == cj: w += W_COMPANY
            if USE_DOMAIN:
                di = df_nodes.loc[i,"domain"]
                dj = df_nodes.loc[j,"domain"]
                if di and di == dj and di not in {"gmail.com","yahoo.com","outlook.com","hotmail.com","icloud.com"}:
                    w += W_DOMAIN
            if sims is not None and sims[i, j] >= POSITION_THRESHOLD:
                w += W_POSITION
            if w > 0:
                G.add_edge(df_nodes.loc[i,"id"], df_nodes.loc[j,"id"], weight=w, kind="inferred")
    return G, df_nodes

def compute_centrality(G):
    H = G.copy()
    if "__YOU__" in H: H.remove_node("__YOU__")
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["name","degree","betweenness"])
    deg = nx.degree_centrality(H)
    btw = nx.betweenness_centrality(H, weight="weight", normalized=True)
    rows = [{"name": H.nodes[n].get("label", n),
             "degree": deg.get(n,0), "betweenness": btw.get(n,0)} for n in H.nodes()]
    return pd.DataFrame(rows).sort_values(["degree","betweenness"], ascending=False)

def compute_partition(G):
    if community_louvain is None:
        return None
    H = G.copy()
    if "__YOU__" in H: H.remove_node("__YOU__")
    if H.number_of_nodes() == 0: return None
    return community_louvain.best_partition(H, weight="weight", random_state=42)

def export_pyvis_html(G, height="650px"):
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    # Faster physics: single solver, fewer bells
    net.barnes_hut(gravity=-18000, spring_length=95, spring_strength=0.01, damping=0.5)
    # No control panel
    # Add nodes/edges
    for n, data in G.nodes(data=True):
        net.add_node(n, label=data.get("label", str(n)), title=data.get("title",""),
                     group=data.get("group",""), size=data.get("size",8))
    for u, v, d in G.edges(data=True):
        color = "#999999" if d.get("kind") == "ego" else "#2b8a3e"
        net.add_edge(u, v, value=float(d.get("weight",1.0)), color=color)
    html = net.generate_html()
    return html

def find_connectors(G, df_nodes, company="", domain="", position="", keywords="", top_k=15):
    texts = (df_nodes["company"].fillna("") + " " + df_nodes["position"].fillna("")).astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    X = vec.fit_transform(texts)
    target = " ".join([t for t in [company, position, keywords] if t])
    sims = cosine_similarity(X, vec.transform([target])).flatten() if target.strip() else np.zeros(len(texts))

    boosts = np.zeros(len(texts))
    if company.strip():
        boosts += (df_nodes["company"].fillna("").str.strip().str.lower() == company.strip().lower()).to_numpy() * 0.5
    if domain.strip():
        boosts += (df_nodes["domain"].fillna("").str.strip().str.lower() == domain.strip().lower()).to_numpy() * 0.5

    # Simple degree blend
    H = G.copy()
    if "__YOU__" in H: H.remove_node("__YOU__")
    deg = nx.degree_centrality(H) if H.number_of_nodes() else {}
    deg_map = {n: deg.get(n,0.0) for n in H.nodes()}
    deg_vec = np.array([deg_map.get(nid,0.0) for nid in df_nodes["id"].tolist()])

    score = 0.75*sims + 0.25*deg_vec
    order = np.argsort(-score)[:top_k]
    out = []
    for idx in order:
        r = df_nodes.iloc[idx]
        out.append({
            "name": r["name"], "company": r["company"], "position": r["position"],
            "email": r["email"], "domain": r["domain"], "profile": r["url"],
            "match_score": round(float(score[idx]), 4)
        })
    return pd.DataFrame(out)

# ---------- UI ----------
st.title("🔗 LinkedIn Network Explorer — Simple")

uploaded = st.file_uploader("Upload your LinkedIn connections.csv", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding_errors="ignore")

    cols = guess_columns(df_raw)
    G, df_nodes = build_graph(df_raw, cols)

    # Optional sampling for performance
    MAX_NODES = 800  # adjustable constant
    H = G.copy()
    if H.number_of_nodes() > MAX_NODES:
        others = [n for n in H.nodes() if n != "__YOU__"]
        keep = set(np.random.choice(others, size=MAX_NODES-1, replace=False))
        keep.add("__YOU__")
        H = H.subgraph(keep).copy()

    # Communities (coloring only; quiet if library missing)
    part = compute_partition(G)
    if part:
        for n in H.nodes():
            if n == "__YOU__": 
                H.nodes[n]["group"] = "You"
            else:
                H.nodes[n]["group"] = f"C{part.get(n,-1)}"

    # --- 1) Network ---
    st.subheader("Network")
    html = export_pyvis_html(H, height="650px")
    st.components.v1.html(html, height=700, scrolling=True)

    # --- 2) Super connectors ---
    st.subheader("Super connectors")
    central = compute_centrality(G)
    st.dataframe(central.head(25), use_container_width=True)

    # --- 3) Who can help you reach... ---
    st.subheader("Find connectors to reach a target")
    with st.form("reach"):
        company = st.text_input("Target company (optional)", value="")
        domain  = st.text_input("Target email domain (optional)", value="")
        role    = st.text_input("Target position / role (optional)", value="")
        keywords = st.text_area("Additional keywords (optional)", value="")
        submitted = st.form_submit_button("Suggest connectors")
    if submitted:
        results = find_connectors(G, df_nodes, company=company, domain=domain, position=role, keywords=keywords, top_k=15)
        st.dataframe(results, use_container_width=True)
else:
    st.info("Upload a CSV to begin.")
