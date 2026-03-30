"""
KG-Enhanced Movie Recommendation — Interactive Demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config & Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KG-Enhanced Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        text-align: center;
    }
    .metric-card h3 {
        color: #a78bfa;
        font-size: 14px;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #e2e8f0;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        border-color: #a78bfa;
    }
    .movie-card .title { color: #e2e8f0; font-weight: 600; font-size: 15px; }
    .movie-card .genre { color: #94a3b8; font-size: 12px; margin-top: 4px; }
    .movie-card .score { color: #22d3ee; font-size: 13px; font-weight: 500; margin-top: 4px; }
    .hit-tag {
        display: inline-block; background: #22c55e; color: #fff;
        font-size: 11px; padding: 2px 8px; border-radius: 4px; margin-left: 6px;
    }
    .miss-tag {
        display: inline-block; background: #ef4444; color: #fff;
        font-size: 11px; padding: 2px 8px; border-radius: 4px; margin-left: 6px;
    }
    .section-header {
        background: linear-gradient(90deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 16px; font-weight: 700; margin: 16px 0 8px 0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .kg-feat-bar {
        background: #1e1e2f; border-radius: 6px; padding: 10px 14px;
        margin: 4px 0; border-left: 3px solid #a78bfa;
    }
    .kg-feat-bar .label { color: #94a3b8; font-size: 13px; }
    .kg-feat-bar .value { color: #e2e8f0; font-size: 16px; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_movies():
    return pd.read_csv("data/processed/movies.csv")

@st.cache_data
def load_tmdb():
    path = "data/tmdb/tmdb_metadata.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_train():
    return pd.read_csv("data/processed/train.csv")

@st.cache_data
def load_test():
    return pd.read_csv("data/processed/test.csv")

@st.cache_resource
def load_kg():
    path = "data/kg/kg_graph.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_triples():
    path = "data/kg/triples.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_scores(model):
    paths = {
        "Item-CF": "results/cf_scores.csv",
        "BPR-MF": "results/mf_scores.csv",
        "LightGCN": "results/lightgcn_scores.csv",
    }
    p = paths.get(model)
    return pd.read_csv(p) if p and os.path.exists(p) else None

@st.cache_data
def load_baseline_results():
    path = "results/baseline_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_ablation_results():
    path = "results/ablation_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance():
    path = "results/feature_importance.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_per_user_ndcg():
    path = "results/per_user_ndcg.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_longtail_analysis():
    path = "results/longtail_analysis.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def metric_card(label, value):
    return f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🎬 KG-Rec Demo")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔗 KG Explorer", "🎯 Recommendations", "🧠 KG Explanation", "📊 Experiments"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("KG-Enhanced Movie Recommendation System")
st.sidebar.caption("Phase 1 — RQ1 & RQ2")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: KG Explorer
# ═══════════════════════════════════════════════════════════════════════════
if page == "🔗 KG Explorer":
    st.markdown("# 🔗 Knowledge Graph Explorer")

    movies_df = load_movies()
    tmdb_df = load_tmdb()
    G = load_kg()
    triples_df = load_triples()

    if G is None:
        st.error("KG graph not found. Run `python run_all.py --phase 2` first.")
        st.stop()

    # --- KG Statistics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Nodes", f"{G.number_of_nodes():,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Edges", f"{G.number_of_edges():,}"), unsafe_allow_html=True)
    with col3:
        n_rel = triples_df["relation"].nunique() if triples_df is not None else "N/A"
        st.markdown(metric_card("Relation Types", n_rel), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Movies", f"{len(movies_df):,}"), unsafe_allow_html=True)

    # --- Relation distribution ---
    if triples_df is not None:
        st.markdown("### Relation Distribution")
        rel_counts = triples_df["relation"].value_counts().reset_index()
        rel_counts.columns = ["Relation", "Count"]
        color_map = {
            "co_liked": "#22c55e",
            "acted_by": "#f97316",
            "has_genre": "#a78bfa",
            "directed_by": "#22d3ee",
            "released_in_decade": "#facc15",
        }
        fig_rel = px.bar(
            rel_counts, x="Relation", y="Count",
            color="Relation", color_discrete_map=color_map,
            template="plotly_dark",
        )
        fig_rel.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=280,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_rel, use_container_width=True)

    st.markdown("---")

    # --- Movie selector + KG subgraph ---
    st.markdown("### Explore Movie Subgraph")
    movie_options = movies_df.sort_values("movie_id")
    movie_labels = {
        row["movie_id"]: f"{row['title']} (ID: {row['movie_id']})"
        for _, row in movie_options.iterrows()
    }

    col_sel, col_depth = st.columns([3, 1])
    with col_sel:
        selected_mid = st.selectbox(
            "Select a movie", options=list(movie_labels.keys()),
            format_func=lambda x: movie_labels[x], index=0,
        )
    with col_depth:
        depth = st.slider("Hop depth", 1, 2, 1)

    movie_node = f"movie_{selected_mid}"

    if movie_node in G:
        # BFS to collect subgraph
        visited = {movie_node}
        frontier = {movie_node}
        sub_edges = []
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    edge_data = G.edges[node, neighbor]
                    sub_edges.append((node, neighbor, edge_data.get("relation", "")))
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        # Limit nodes for performance
        MAX_NODES = 80
        if len(visited) > MAX_NODES:
            direct = {movie_node}
            for n in G.neighbors(movie_node):
                direct.add(n)
            remaining = list(visited - direct)
            np.random.seed(42)
            sampled = set(np.random.choice(remaining, MAX_NODES - len(direct), replace=False))
            visited = direct | sampled
            sub_edges = [(a, b, r) for a, b, r in sub_edges if a in visited and b in visited]

        # Build plotly network graph
        node_colors = {
            "movie": "#6366f1", "actor": "#f97316", "director": "#22d3ee",
            "genre": "#a78bfa", "decade": "#facc15", "other": "#64748b",
        }

        def get_node_type(n):
            for prefix in ["movie_", "actor_", "director_", "genre_", "decade_"]:
                if n.startswith(prefix):
                    return prefix.rstrip("_")
            return "other"

        def get_node_label(n):
            parts = n.split("_", 1)
            return parts[1] if len(parts) > 1 else n

        # Spring layout
        sub_G = nx.Graph()
        for a, b, r in sub_edges:
            sub_G.add_edge(a, b)
        pos = nx.spring_layout(sub_G, seed=42, k=2.0/np.sqrt(len(visited)))

        edge_x, edge_y = [], []
        for a, b, r in sub_edges:
            if a in pos and b in pos:
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n in visited:
            if n not in pos:
                continue
            x, y = pos[n]
            ntype = get_node_type(n)
            label = get_node_label(n)
            if ntype == "movie":
                mid = int(n.split("_")[1])
                mrow = movies_df[movies_df["movie_id"] == mid]
                if len(mrow) > 0:
                    label = mrow.iloc[0]["title"]
            node_x.append(x)
            node_y.append(y)
            node_text.append(label)
            is_selected = (n == movie_node)
            node_color.append("#facc15" if is_selected else node_colors.get(ntype, "#64748b"))
            node_size.append(18 if is_selected else 10)

        fig_graph = go.Figure()
        fig_graph.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#475569"),
            hoverinfo="none",
        ))
        fig_graph.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#1e1e2f")),
            text=node_text, textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            hoverinfo="text",
        ))
        fig_graph.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        col_graph, col_info = st.columns([3, 1])
        with col_graph:
            st.plotly_chart(fig_graph, use_container_width=True)
        with col_info:
            st.markdown("**Legend**")
            for ntype, color in node_colors.items():
                if ntype != "other":
                    st.markdown(f'<span style="color:{color}">⬤</span> {ntype.capitalize()}', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#facc15">⬤</span> Selected', unsafe_allow_html=True)
            st.markdown("---")

            movie_row = movies_df[movies_df["movie_id"] == selected_mid]
            if len(movie_row) > 0:
                row = movie_row.iloc[0]
                st.markdown(f"**{row['title']}**")
                st.markdown(f"Genres: `{row.get('genres', 'N/A')}`")
                if pd.notna(row.get("year")):
                    st.markdown(f"Year: {int(row['year'])}")

            if tmdb_df is not None:
                tmdb_row = tmdb_df[tmdb_df["movie_id"] == selected_mid]
                if len(tmdb_row) > 0:
                    tr = tmdb_row.iloc[0]
                    if pd.notna(tr.get("vote_average")):
                        st.markdown(f"TMDB: ⭐ {tr['vote_average']:.1f} ({int(tr.get('vote_count', 0))} votes)")
                    if pd.notna(tr.get("overview")) and str(tr["overview"]) != "nan":
                        st.markdown("---")
                        overview = str(tr["overview"])
                        st.markdown(f"*{overview[:300]}...*" if len(overview) > 300 else f"*{overview}*")

            st.markdown("---")
            st.markdown(f"Subgraph: **{len(visited)}** nodes, **{len(sub_edges)}** edges")
    else:
        st.warning(f"Movie ID {selected_mid} not found in KG.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Recommendations
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.markdown("# 🎯 Recommendation Comparison")

    train_df = load_train()
    test_df = load_test()
    movies_df = load_movies()

    # Build user history
    user_history = train_df.groupby("user_id").apply(
        lambda g: g.sort_values("timestamp").tail(20)[["movie_id"]].values.tolist()
    )

    # User selector
    test_users = sorted(test_df["user_id"].unique())
    selected_user = st.selectbox("Select User ID", test_users[:200],
                                  format_func=lambda x: f"User {x}")

    # User profile
    if selected_user in user_history.index:
        history = user_history[selected_user]
        st.markdown(f'<div class="section-header">User {selected_user} — Watch History (last 20)</div>', unsafe_allow_html=True)

        hist_cols = st.columns(min(len(history), 6))
        for i, (mid_arr,) in enumerate(history[:6]):
            mid = int(mid_arr)
            mrow = movies_df[movies_df["movie_id"] == mid]
            title = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {mid}"
            genres = mrow.iloc[0].get("genres", "") if len(mrow) > 0 else ""
            with hist_cols[i]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="title">{title}</div>
                    <div class="genre">{genres}</div>
                </div>
                """, unsafe_allow_html=True)

        if len(history) > 6:
            st.caption(f"... and {len(history) - 6} more movies")

    st.markdown("---")

    # Ground truth
    gt_items = set(test_df[test_df["user_id"] == selected_user]["movie_id"].tolist())

    # Load and compare models
    st.markdown(f'<div class="section-header">Top-10 Recommendations — 3 Models Side by Side</div>', unsafe_allow_html=True)

    model_names = ["Item-CF", "BPR-MF", "LightGCN"]
    model_cols = st.columns(3)

    for idx, model_name in enumerate(model_names):
        scores_df = load_scores(model_name)
        with model_cols[idx]:
            st.markdown(f"#### {model_name}")
            if scores_df is None:
                st.warning("Scores not available")
                continue

            user_scores = scores_df[scores_df["user_id"] == selected_user]
            if len(user_scores) == 0:
                st.info("No predictions for this user")
                continue

            score_col = [c for c in user_scores.columns if c not in ["user_id", "movie_id"]][0]
            top10 = user_scores.nlargest(10, score_col)

            hits = 0
            for rank, (_, row) in enumerate(top10.iterrows(), 1):
                mid = int(row["movie_id"])
                mrow = movies_df[movies_df["movie_id"] == mid]
                title = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {mid}"
                genres = mrow.iloc[0].get("genres", "") if len(mrow) > 0 else ""
                is_hit = mid in gt_items
                if is_hit:
                    hits += 1
                tag = '<span class="hit-tag">HIT</span>' if is_hit else ""

                st.markdown(f"""
                <div class="movie-card">
                    <div class="title">{rank}. {title} {tag}</div>
                    <div class="genre">{genres}</div>
                    <div class="score">Score: {row[score_col]:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Hits: {hits}/10** | Ground truth items: {len(gt_items)}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: KG Explanation
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠 KG Explanation":
    st.markdown("# 🧠 KG-based Recommendation Explanation")

    G = load_kg()
    train_df = load_train()
    movies_df = load_movies()
    tmdb_df = load_tmdb()

    if G is None:
        st.error("KG not found.")
        st.stop()

    col1, col2 = st.columns(2)
    test_users = sorted(load_test()["user_id"].unique())

    with col1:
        selected_user = st.selectbox("Select User", test_users[:200],
                                      format_func=lambda x: f"User {x}", key="kg_user")
    with col2:
        cf_scores = load_scores("Item-CF")
        candidate_mids = []
        if cf_scores is not None:
            user_cf = cf_scores[cf_scores["user_id"] == selected_user].nlargest(20, "cf_score")
            candidate_mids = user_cf["movie_id"].tolist()

        if candidate_mids:
            movie_labels = {}
            for mid in candidate_mids:
                mrow = movies_df[movies_df["movie_id"] == int(mid)]
                movie_labels[int(mid)] = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {int(mid)}"
            selected_candidate = st.selectbox(
                "Select Candidate Movie",
                options=list(movie_labels.keys()),
                format_func=lambda x: movie_labels[x],
            )
        else:
            selected_candidate = st.number_input("Candidate Movie ID", value=1, step=1)

    # User history
    user_train = train_df[train_df["user_id"] == selected_user].sort_values("timestamp")
    user_movies = user_train["movie_id"].tolist()[-20:]

    candidate_node = f"movie_{selected_candidate}"

    def get_neighbors_by_rel(G, node, rel):
        result = set()
        if node not in G:
            return result
        for nb in G.neighbors(node):
            if G.edges[node, nb].get("relation") == rel:
                result.add(nb)
        return result

    st.markdown("---")

    # Compute KG connections
    candidate_actors = get_neighbors_by_rel(G, candidate_node, "acted_by")
    candidate_directors = get_neighbors_by_rel(G, candidate_node, "directed_by")
    candidate_genres = get_neighbors_by_rel(G, candidate_node, "has_genre")
    candidate_decade = get_neighbors_by_rel(G, candidate_node, "released_in_decade")
    candidate_coliked = get_neighbors_by_rel(G, candidate_node, "co_liked")

    shared_actors_detail = []
    shared_directors_detail = []
    shared_genres_all = set()
    same_decade_count = 0
    co_liked_movies = []

    for hist_mid in user_movies:
        hist_node = f"movie_{hist_mid}"
        mrow = movies_df[movies_df["movie_id"] == hist_mid]
        hist_title = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {hist_mid}"

        # Shared actors/directors/genres
        for a in candidate_actors & get_neighbors_by_rel(G, hist_node, "acted_by"):
            shared_actors_detail.append((a.split("_", 1)[1], hist_title))
        for d in candidate_directors & get_neighbors_by_rel(G, hist_node, "directed_by"):
            shared_directors_detail.append((d.split("_", 1)[1], hist_title))
        shared_genres_all |= {g.split("_", 1)[1] for g in candidate_genres & get_neighbors_by_rel(G, hist_node, "has_genre")}

        # Same decade
        if candidate_decade & get_neighbors_by_rel(G, hist_node, "released_in_decade"):
            same_decade_count += 1

        # Co-liked
        if hist_node in candidate_coliked:
            co_liked_movies.append(hist_title)

    # Display explanation
    cand_row = movies_df[movies_df["movie_id"] == selected_candidate]
    cand_title = cand_row.iloc[0]["title"] if len(cand_row) > 0 else f"Movie {selected_candidate}"

    st.markdown(f"### Why recommend **{cand_title}** to User {selected_user}?")

    # Feature cards — 6 KG signals
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
    unique_actors = set(a for a, _ in shared_actors_detail)
    unique_dirs = set(d for d, _ in shared_directors_detail)
    with fc1:
        st.markdown(metric_card("Shared Actors", len(unique_actors)), unsafe_allow_html=True)
    with fc2:
        st.markdown(metric_card("Shared Directors", len(unique_dirs)), unsafe_allow_html=True)
    with fc3:
        st.markdown(metric_card("Shared Genres", len(shared_genres_all)), unsafe_allow_html=True)
    with fc4:
        decade_ratio = f"{same_decade_count}/{len(user_movies)}"
        st.markdown(metric_card("Same Decade", decade_ratio), unsafe_allow_html=True)
    with fc5:
        st.markdown(metric_card("Co-liked", len(co_liked_movies)), unsafe_allow_html=True)
    with fc6:
        best_path_len = None
        for hist_mid in user_movies[:10]:
            hist_node = f"movie_{hist_mid}"
            if hist_node in G and candidate_node in G:
                try:
                    path = nx.shortest_path(G, hist_node, candidate_node)
                    plen = len(path) - 1
                    if best_path_len is None or plen < best_path_len:
                        best_path_len = plen
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        st.markdown(metric_card("Min Path", f"{best_path_len or '∞'} hops"), unsafe_allow_html=True)

    st.markdown("---")

    # Detailed explanations
    if shared_actors_detail:
        st.markdown("#### 🎭 Shared Actors")
        for actor, hist_title in shared_actors_detail[:8]:
            st.markdown(f"""
            <div class="kg-feat-bar">
                <span class="label">Actor</span>
                <span class="value" style="margin-left:8px;">{actor}</span>
                <span class="label" style="margin-left:16px;">also in</span>
                <span style="color:#a78bfa; margin-left:8px;">{hist_title}</span>
            </div>
            """, unsafe_allow_html=True)

    if shared_directors_detail:
        st.markdown("#### 🎬 Shared Directors")
        for director, hist_title in shared_directors_detail[:5]:
            st.markdown(f"""
            <div class="kg-feat-bar">
                <span class="label">Director</span>
                <span class="value" style="margin-left:8px;">{director}</span>
                <span class="label" style="margin-left:16px;">also directed</span>
                <span style="color:#22d3ee; margin-left:8px;">{hist_title}</span>
            </div>
            """, unsafe_allow_html=True)

    if co_liked_movies:
        st.markdown("#### 🤝 Co-liked Movies")
        st.caption("Movies that many users liked together with this candidate")
        for title in co_liked_movies[:6]:
            st.markdown(f"""
            <div class="kg-feat-bar">
                <span class="value">{title}</span>
                <span class="label" style="margin-left:12px;">co-liked with {cand_title}</span>
            </div>
            """, unsafe_allow_html=True)

    if shared_genres_all:
        st.markdown("#### 🏷️ Shared Genres")
        genre_tags = " ".join(
            f'<span style="background:#a78bfa22; border:1px solid #a78bfa; color:#a78bfa; '
            f'padding:4px 12px; border-radius:20px; margin:2px; display:inline-block; '
            f'font-size:13px;">{g}</span>'
            for g in sorted(shared_genres_all)
        )
        st.markdown(genre_tags, unsafe_allow_html=True)

    if not shared_actors_detail and not shared_directors_detail and not shared_genres_all and not co_liked_movies:
        st.info("No direct KG connections found between this candidate and user's history. "
                "The recommendation relies purely on collaborative filtering signals.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Experiments Dashboard
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Experiments":
    st.markdown("# 📊 Experiment Dashboard")

    baseline_results = load_baseline_results()
    ablation_results = load_ablation_results()
    feat_importance = load_feature_importance()
    per_user_ndcg = load_per_user_ndcg()
    longtail = load_longtail_analysis()

    # --- Tab layout ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Recall Baselines", "🔬 Ablation Study",
        "🌿 Feature Importance", "📊 Long-tail Analysis",
    ])

    # --- Tab 1: Baselines ---
    with tab1:
        if baseline_results:
            st.markdown("### Recall Model Comparison")
            st.caption("Evaluated on full catalog (~3,125 movies), Top-10")

            metrics = ["Hit@10", "NDCG@10", "Recall@10", "MRR"]
            models = list(baseline_results.keys())

            fig_base = go.Figure()
            colors = ["#6366f1", "#a78bfa", "#22d3ee"]
            for i, model in enumerate(models):
                fig_base.add_trace(go.Bar(
                    name=model,
                    x=metrics,
                    y=[baseline_results[model].get(m, 0) for m in metrics],
                    marker_color=colors[i % len(colors)],
                    text=[f"{baseline_results[model].get(m, 0):.4f}" for m in metrics],
                    textposition="outside", textfont=dict(size=11),
                ))
            fig_base.update_layout(
                barmode="group", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=1.12),
                yaxis=dict(gridcolor="#2a2a4a"),
            )
            st.plotly_chart(fig_base, use_container_width=True)

            # Coverage comparison
            st.markdown("### Coverage (Catalog Diversity)")
            cov_data = pd.DataFrame([
                {"Model": m, "Coverage": baseline_results[m].get("Coverage", 0)}
                for m in models
            ])
            fig_cov = px.bar(cov_data, x="Model", y="Coverage", color="Model",
                             color_discrete_sequence=colors, template="plotly_dark",
                             text=cov_data["Coverage"].apply(lambda x: f"{x:.1%}"))
            fig_cov.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=280,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_cov, use_container_width=True)

            # Per-user NDCG distribution
            if per_user_ndcg:
                st.markdown("### Per-User NDCG@10 Distribution")
                fig_dist = go.Figure()
                dist_colors = {"Item-CF": "#6366f1", "MF-64": "#a78bfa", "LightGCN": "#22d3ee"}
                for model_name, user_ndcgs in per_user_ndcg.items():
                    values = list(user_ndcgs.values())
                    fig_dist.add_trace(go.Violin(
                        y=values, name=model_name,
                        box_visible=True, meanline_visible=True,
                        fillcolor=dist_colors.get(model_name, "#6366f1"),
                        opacity=0.7, line_color="#e2e8f0",
                    ))
                fig_dist.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=400, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title="NDCG@10", gridcolor="#2a2a4a"),
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Baseline results not found. Run `python run_all.py --phase 1`.")

    # --- Tab 2: Ablation ---
    with tab2:
        if ablation_results:
            st.markdown("### Ablation Study: KG Feature Contribution (RQ1)")
            st.caption("Evaluated on recall top-100 candidates (distribution-matched)")

            method = st.radio("Ranking Method", ["Pointwise", "LambdaMART"], horizontal=True)

            # Filter variants for selected method
            recall_only_key = "Recall-only"
            variant_keys = [k for k in ablation_results if f"[{method}]" in k]
            display_keys = [recall_only_key] + variant_keys if recall_only_key in ablation_results else variant_keys
            display_names = []
            for k in display_keys:
                name = k.replace(f" [{method}]", "") if f" [{method}]" in k else k
                display_names.append(name)

            metrics_ab = ["NDCG@10", "Recall@10", "Hit@10", "MRR@10"]

            # Bar chart
            ab_colors = {
                "Recall-only": "#475569",
                "V1 (CF)": "#64748b",
                "V2 (CF+Content)": "#a78bfa",
                "V3 (CF+Content+KG)": "#22d3ee",
                "V3e (CF+Content+KGEmb)": "#f97316",
                "V4 (CF+Content+KG+Emb)": "#22c55e",
            }
            fig_ab = go.Figure()
            for key, name in zip(display_keys, display_names):
                vals = ablation_results[key]
                fig_ab.add_trace(go.Bar(
                    name=name, x=metrics_ab,
                    y=[vals.get(m, 0) for m in metrics_ab],
                    marker_color=ab_colors.get(name, "#6366f1"),
                    text=[f"{vals.get(m, 0):.4f}" for m in metrics_ab],
                    textposition="outside", textfont=dict(size=10),
                ))
            fig_ab.update_layout(
                barmode="group", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=420, margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=1.15),
                yaxis=dict(gridcolor="#2a2a4a"),
            )
            st.plotly_chart(fig_ab, use_container_width=True)

            # Summary table
            st.markdown("### Results Table")
            rows = []
            for key, name in zip(display_keys, display_names):
                vals = ablation_results[key]
                rows.append({
                    "Variant": name,
                    "NDCG@10": f"{vals.get('NDCG@10', 0):.4f}",
                    "Recall@10": f"{vals.get('Recall@10', 0):.4f}",
                    "Hit@10": f"{vals.get('Hit@10', 0):.4f}",
                    "MRR@10": f"{vals.get('MRR@10', 0):.4f}",
                    "Coverage": f"{vals.get('Coverage', 0):.1%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Ablation results not found. Run `python run_all.py --phase 3`.")

    # --- Tab 3: Feature Importance ---
    with tab3:
        if feat_importance:
            st.markdown("### Feature Importance Analysis")

            method = st.radio("Method", ["Pointwise", "LambdaMART"], horizontal=True, key="fi_method")
            variant = st.selectbox("Variant", [
                k for k in feat_importance if f"[{method}]" in k
            ])

            if variant and variant in feat_importance:
                imp = feat_importance[variant]
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v} for k, v in imp.items()
                ]).sort_values("Importance", ascending=True)

                # Categorize features
                def categorize(f):
                    if f.startswith("kg_"):
                        return "KG Feature"
                    elif f in ("cf_score", "kg_recall_score"):
                        return "CF Feature"
                    else:
                        return "Content Feature"

                imp_df["Type"] = imp_df["Feature"].apply(categorize)
                total = imp_df["Importance"].sum()
                imp_df["Share"] = imp_df["Importance"] / total * 100

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    color="Type",
                    color_discrete_map={
                        "KG Feature": "#22d3ee",
                        "CF Feature": "#6366f1",
                        "Content Feature": "#a78bfa",
                    },
                    template="plotly_dark",
                    hover_data={"Share": ":.1f"},
                )
                fig_imp.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=max(350, len(imp_df) * 30),
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#2a2a4a", title="Gain"),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                # Share summary
                share_df = imp_df.groupby("Type")["Importance"].sum().reset_index()
                share_df["Share"] = share_df["Importance"] / total * 100
                share_df = share_df.sort_values("Share", ascending=False)

                cols = st.columns(len(share_df))
                for i, (_, row) in enumerate(share_df.iterrows()):
                    with cols[i]:
                        st.markdown(metric_card(row["Type"], f"{row['Share']:.1f}%"), unsafe_allow_html=True)
        else:
            st.warning("Feature importance not found. Run `python run_all.py --phase 3`.")

    # --- Tab 4: Long-tail Analysis ---
    with tab4:
        if longtail:
            st.markdown("### Long-tail Analysis (RQ2)")
            st.caption("Does KG disproportionately help long-tail items?")

            # Head/Tail stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(metric_card("Head Movies", longtail["head_movies_count"]), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Tail Movies", longtail["tail_movies_count"]), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("Threshold", f"≤ {longtail['head_tail_threshold']} interactions"), unsafe_allow_html=True)

            # Stratified recall chart
            strat = longtail.get("stratified_recall", {})
            if strat:
                st.markdown("### Head vs Tail Recall@10")
                rows = []
                for variant, vals in strat.items():
                    rows.append({
                        "Variant": variant,
                        "Head Recall@10": vals.get("head_recall", 0),
                        "Tail Recall@10": vals.get("tail_recall", 0),
                    })
                strat_df = pd.DataFrame(rows)

                fig_strat = go.Figure()
                fig_strat.add_trace(go.Bar(
                    name="Head", x=strat_df["Variant"], y=strat_df["Head Recall@10"],
                    marker_color="#6366f1",
                    text=strat_df["Head Recall@10"].apply(lambda x: f"{x:.4f}"),
                    textposition="outside",
                ))
                fig_strat.add_trace(go.Bar(
                    name="Tail", x=strat_df["Variant"], y=strat_df["Tail Recall@10"],
                    marker_color="#f97316",
                    text=strat_df["Tail Recall@10"].apply(lambda x: f"{x:.4f}"),
                    textposition="outside",
                ))
                fig_strat.update_layout(
                    barmode="group", template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=400, margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=1.12),
                    yaxis=dict(gridcolor="#2a2a4a", title="Recall@10"),
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_strat, use_container_width=True)

            # Entropy analysis
            entropy = longtail.get("entropy_results", {})
            if entropy:
                st.markdown("### User Genre Entropy Analysis")
                st.caption("KG benefit by user preference concentration: low entropy = focused interests")

                ent_rows = []
                for variant, vals in entropy.items():
                    for bucket in ["low", "mid", "high"]:
                        ent_rows.append({
                            "Variant": variant,
                            "Entropy": bucket.capitalize(),
                            "Recall@10": vals.get(f"{bucket}_recall", 0),
                        })
                ent_df = pd.DataFrame(ent_rows)

                fig_ent = px.bar(
                    ent_df, x="Variant", y="Recall@10", color="Entropy",
                    barmode="group",
                    color_discrete_map={"Low": "#22d3ee", "Mid": "#a78bfa", "High": "#f97316"},
                    template="plotly_dark",
                    text=ent_df["Recall@10"].apply(lambda x: f"{x:.3f}"),
                )
                fig_ent.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=400, margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=1.12),
                    yaxis=dict(gridcolor="#2a2a4a"),
                    xaxis=dict(tickangle=-30),
                )
                fig_ent.update_traces(textposition="outside", textfont=dict(size=10))
                st.plotly_chart(fig_ent, use_container_width=True)
        else:
            st.warning("Long-tail analysis not found. Run `python run_all.py --phase 4`.")
