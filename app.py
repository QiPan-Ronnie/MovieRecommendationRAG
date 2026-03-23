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
from streamlit_agraph import agraph, Node, Edge, Config

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
    /* Dark card style */
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
    /* Movie card */
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
    .movie-card .title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 15px;
    }
    .movie-card .genre {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }
    .movie-card .score {
        color: #22d3ee;
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
    }
    .hit-tag {
        display: inline-block;
        background: #22c55e;
        color: #fff;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .miss-tag {
        display: inline-block;
        background: #ef4444;
        color: #fff;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 16px;
        font-weight: 700;
        margin: 16px 0 8px 0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    /* KG feature bar */
    .kg-feat-bar {
        background: #1e1e2f;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 3px solid #a78bfa;
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
    df = pd.read_csv("data/processed/movies.csv")
    return df

@st.cache_data
def load_tmdb():
    path = "data/tmdb/tmdb_metadata.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

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
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_scores(model):
    paths = {
        "Item-CF": "results/cf_scores.csv",
        "BPR-MF": "results/mf_scores.csv",
        "LightGCN": "results/lightgcn_scores.csv",
    }
    p = paths.get(model)
    if p and os.path.exists(p):
        return pd.read_csv(p)
    return None

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


def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <p>{value}</p>
    </div>
    """


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
        st.error("KG graph not found. Run `python src/kg/build_kg.py` first.")
        st.stop()

    # --- KG Statistics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Nodes", f"{G.number_of_nodes():,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Edges", f"{G.number_of_edges():,}"), unsafe_allow_html=True)
    with col3:
        if triples_df is not None:
            n_rel = triples_df["relation"].nunique()
        else:
            n_rel = "N/A"
        st.markdown(metric_card("Relation Types", n_rel), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Movies", f"{len(movies_df):,}"), unsafe_allow_html=True)

    # --- Relation distribution ---
    if triples_df is not None:
        st.markdown("### Relation Distribution")
        rel_counts = triples_df["relation"].value_counts().reset_index()
        rel_counts.columns = ["Relation", "Count"]
        color_map = {
            "has_genre": "#a78bfa",
            "acted_by": "#f97316",
            "directed_by": "#22d3ee",
        }
        fig_rel = px.bar(
            rel_counts, x="Relation", y="Count",
            color="Relation",
            color_discrete_map=color_map,
            template="plotly_dark",
        )
        fig_rel.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=280,
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
            "Select a movie",
            options=list(movie_labels.keys()),
            format_func=lambda x: movie_labels[x],
            index=0,
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
            # Keep the selected movie + its direct neighbors + sample the rest
            direct = {movie_node}
            for n in G.neighbors(movie_node):
                direct.add(n)
            remaining = list(visited - direct)
            np.random.seed(42)
            sampled = set(np.random.choice(remaining, MAX_NODES - len(direct), replace=False))
            visited = direct | sampled
            sub_edges = [(a, b, r) for a, b, r in sub_edges if a in visited and b in visited]

        # Build agraph nodes & edges
        node_colors = {
            "movie": "#6366f1",
            "actor": "#f97316",
            "director": "#22d3ee",
            "genre": "#a78bfa",
        }
        node_sizes = {
            "movie": 22,
            "actor": 16,
            "director": 18,
            "genre": 14,
        }

        def get_node_type(n):
            if n.startswith("movie_"): return "movie"
            if n.startswith("actor_"): return "actor"
            if n.startswith("director_"): return "director"
            if n.startswith("genre_"): return "genre"
            return "other"

        def get_node_label(n):
            parts = n.split("_", 1)
            return parts[1] if len(parts) > 1 else n

        nodes = []
        for n in visited:
            ntype = get_node_type(n)
            label = get_node_label(n)
            if n == movie_node:
                # Highlight selected movie
                nodes.append(Node(
                    id=n, label=label, size=30,
                    color="#facc15", font={"color": "#facc15", "size": 14},
                    borderWidth=3, borderWidthSelected=5,
                ))
            else:
                nodes.append(Node(
                    id=n, label=label,
                    size=node_sizes.get(ntype, 14),
                    color=node_colors.get(ntype, "#64748b"),
                    font={"color": "#94a3b8", "size": 11},
                ))

        edge_colors = {
            "has_genre": "#a78bfa",
            "acted_by": "#f97316",
            "directed_by": "#22d3ee",
        }
        edges = []
        seen_edges = set()
        for a, b, r in sub_edges:
            key = tuple(sorted([a, b]))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(Edge(
                    source=a, target=b, label=r,
                    color=edge_colors.get(r, "#475569"),
                    width=1.5,
                ))

        config = Config(
            width="100%", height=500,
            directed=False,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#facc15",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": False},
        )

        col_graph, col_info = st.columns([3, 1])
        with col_graph:
            agraph(nodes=nodes, edges=edges, config=config)
        with col_info:
            # Legend
            st.markdown("**Legend**")
            st.markdown(f'<span style="color:#facc15">⬤</span> Selected Movie', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#6366f1">⬤</span> Movie', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#f97316">⬤</span> Actor', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#22d3ee">⬤</span> Director', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#a78bfa">⬤</span> Genre', unsafe_allow_html=True)
            st.markdown("---")

            # Movie info
            movie_row = movies_df[movies_df["movie_id"] == selected_mid]
            if len(movie_row) > 0:
                row = movie_row.iloc[0]
                st.markdown(f"**{row['title']}**")
                st.markdown(f"Genres: `{row.get('genres', 'N/A')}`")
                if row.get("year"):
                    st.markdown(f"Year: {row['year']}")

            if tmdb_df is not None:
                tmdb_row = tmdb_df[tmdb_df["movie_id"] == selected_mid]
                if len(tmdb_row) > 0:
                    tr = tmdb_row.iloc[0]
                    if pd.notna(tr.get("vote_average")):
                        st.markdown(f"TMDB: ⭐ {tr['vote_average']:.1f} ({int(tr.get('vote_count', 0))} votes)")
                    if pd.notna(tr.get("overview")) and str(tr["overview"]) != "nan":
                        st.markdown("---")
                        st.markdown(f"*{str(tr['overview'])[:300]}...*" if len(str(tr['overview'])) > 300 else f"*{tr['overview']}*")

            # Subgraph stats
            st.markdown("---")
            st.markdown(f"Subgraph: **{len(visited)}** nodes, **{len(edges)}** edges")
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
    tmdb_df = load_tmdb()

    # Build user history
    user_history = train_df.groupby("user_id").apply(
        lambda g: g.sort_values("timestamp").tail(20)[["movie_id", "rating"]].values.tolist()
    )

    # User selector
    test_users = sorted(test_df["user_id"].unique())
    selected_user = st.selectbox("Select User ID", test_users[:200],
                                  format_func=lambda x: f"User {x}")

    # User profile
    if selected_user in user_history.index:
        history = user_history[selected_user]
        st.markdown(f'<div class="section-header">User {selected_user} — Watch History (rating ≥ 4, last 20)</div>', unsafe_allow_html=True)

        hist_cols = st.columns(min(len(history), 6))
        for i, (mid, rating) in enumerate(history[:6]):
            mid = int(mid)
            mrow = movies_df[movies_df["movie_id"] == mid]
            title = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {mid}"
            genres = mrow.iloc[0].get("genres", "") if len(mrow) > 0 else ""
            with hist_cols[i]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="title">{title}</div>
                    <div class="genre">{genres}</div>
                    <div class="score">★ {int(rating)}</div>
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
        # Get this user's recommended movies from Item-CF
        cf_scores = load_scores("Item-CF")
        candidate_mids = []
        if cf_scores is not None:
            user_cf = cf_scores[cf_scores["user_id"] == selected_user].nlargest(20, "cf_score")
            candidate_mids = user_cf["movie_id"].tolist()

        if candidate_mids:
            movie_labels = {}
            for mid in candidate_mids:
                mrow = movies_df[movies_df["movie_id"] == int(mid)]
                if len(mrow) > 0:
                    movie_labels[int(mid)] = mrow.iloc[0]["title"]
                else:
                    movie_labels[int(mid)] = f"Movie {int(mid)}"
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

    shared_actors_detail = []
    shared_directors_detail = []
    shared_genres_all = set()
    connected_movies = []

    for hist_mid in user_movies:
        hist_node = f"movie_{hist_mid}"
        hist_actors = get_neighbors_by_rel(G, hist_node, "acted_by")
        hist_directors = get_neighbors_by_rel(G, hist_node, "directed_by")
        hist_genres = get_neighbors_by_rel(G, hist_node, "has_genre")

        common_actors = candidate_actors & hist_actors
        common_directors = candidate_directors & hist_directors
        common_genres = candidate_genres & hist_genres

        mrow = movies_df[movies_df["movie_id"] == hist_mid]
        hist_title = mrow.iloc[0]["title"] if len(mrow) > 0 else f"Movie {hist_mid}"

        for a in common_actors:
            shared_actors_detail.append((a.split("_", 1)[1], hist_title, hist_mid))
        for d in common_directors:
            shared_directors_detail.append((d.split("_", 1)[1], hist_title, hist_mid))
        shared_genres_all |= {g.split("_", 1)[1] for g in common_genres}

        if common_actors or common_directors or common_genres:
            connected_movies.append(hist_mid)

    # Display explanation
    cand_row = movies_df[movies_df["movie_id"] == selected_candidate]
    cand_title = cand_row.iloc[0]["title"] if len(cand_row) > 0 else f"Movie {selected_candidate}"

    st.markdown(f"### Why recommend **{cand_title}** to User {selected_user}?")

    # Feature cards
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        unique_actors = set(a for a, _, _ in shared_actors_detail)
        st.markdown(metric_card("Shared Actors", len(unique_actors)), unsafe_allow_html=True)
    with fc2:
        unique_dirs = set(d for d, _, _ in shared_directors_detail)
        st.markdown(metric_card("Shared Directors", len(unique_dirs)), unsafe_allow_html=True)
    with fc3:
        st.markdown(metric_card("Shared Genres", len(shared_genres_all)), unsafe_allow_html=True)
    with fc4:
        # Shortest path
        best_path_len = None
        best_path = None
        for hist_mid in user_movies[:10]:
            hist_node = f"movie_{hist_mid}"
            if hist_node in G and candidate_node in G:
                try:
                    path = nx.shortest_path(G, hist_node, candidate_node)
                    if best_path is None or len(path) < len(best_path):
                        best_path = path
                        best_path_len = len(path) - 1
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        st.markdown(metric_card("Shortest Path", f"{best_path_len or '∞'} hops"), unsafe_allow_html=True)

    st.markdown("---")

    # Detailed explanations
    if shared_actors_detail:
        st.markdown("#### 🎭 Shared Actor Connections")
        for actor, hist_title, _ in shared_actors_detail[:8]:
            st.markdown(f"""
            <div class="kg-feat-bar">
                <span class="label">Actor</span>
                <span class="value" style="margin-left:8px;">{actor}</span>
                <span class="label" style="margin-left:16px;">also in</span>
                <span style="color:#a78bfa; margin-left:8px;">{hist_title}</span>
            </div>
            """, unsafe_allow_html=True)

    if shared_directors_detail:
        st.markdown("#### 🎬 Shared Director Connections")
        for director, hist_title, _ in shared_directors_detail[:5]:
            st.markdown(f"""
            <div class="kg-feat-bar">
                <span class="label">Director</span>
                <span class="value" style="margin-left:8px;">{director}</span>
                <span class="label" style="margin-left:16px;">also directed</span>
                <span style="color:#22d3ee; margin-left:8px;">{hist_title}</span>
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

    # Path visualization
    if best_path and len(best_path) > 1:
        st.markdown("---")
        st.markdown("#### 🔗 Shortest KG Path")

        path_nodes = []
        path_edges = []
        for n in best_path:
            ntype = "movie" if n.startswith("movie_") else \
                    "actor" if n.startswith("actor_") else \
                    "director" if n.startswith("director_") else "genre"
            label = n.split("_", 1)[1]
            # Lookup movie title
            if ntype == "movie":
                mid = int(n.split("_")[1])
                mrow = movies_df[movies_df["movie_id"] == mid]
                if len(mrow) > 0:
                    label = mrow.iloc[0]["title"]
            color = {"movie": "#6366f1", "actor": "#f97316",
                     "director": "#22d3ee", "genre": "#a78bfa"}.get(ntype, "#64748b")
            is_endpoint = (n == best_path[0] or n == best_path[-1])
            path_nodes.append(Node(
                id=n, label=label,
                size=28 if is_endpoint else 20,
                color="#facc15" if is_endpoint else color,
                font={"color": "#e2e8f0", "size": 13},
            ))

        for i in range(len(best_path) - 1):
            a, b = best_path[i], best_path[i+1]
            rel = G.edges[a, b].get("relation", "") if G.has_edge(a, b) else ""
            path_edges.append(Edge(source=a, target=b, label=rel, color="#94a3b8", width=2))

        config = Config(width="100%", height=200, directed=False, physics=False,
                        hierarchical=True, hierarchicalSortMethod="directed")
        agraph(nodes=path_nodes, edges=path_edges, config=config)

    if not shared_actors_detail and not shared_directors_detail and not shared_genres_all:
        st.info("No direct KG connections found between this candidate and user's history. "
                "This is a case where KG features would be zero — the recommendation "
                "relies purely on collaborative filtering signals.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Experiments Dashboard
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Experiments":
    st.markdown("# 📊 Experiment Dashboard")

    baseline_results = load_baseline_results()
    ablation_results = load_ablation_results()
    feat_importance = load_feature_importance()
    per_user_ndcg = load_per_user_ndcg()

    # --- Baseline comparison ---
    if baseline_results:
        st.markdown("### Baseline Model Comparison")
        st.caption("Evaluated on full catalog, Top-10")

        metrics = ["Hit@10", "NDCG@10", "Recall@10", "MRR", "Coverage"]
        models = list(baseline_results.keys())

        # Metric cards
        cols = st.columns(len(models))
        for i, model in enumerate(models):
            with cols[i]:
                st.markdown(f"**{model}**")
                for m in metrics[:3]:
                    val = baseline_results[model].get(m, 0)
                    st.markdown(metric_card(m, f"{val:.4f}"), unsafe_allow_html=True)

        # Bar chart
        fig_base = go.Figure()
        colors = ["#6366f1", "#a78bfa", "#22d3ee"]
        for i, model in enumerate(models):
            fig_base.add_trace(go.Bar(
                name=model,
                x=metrics[:4],
                y=[baseline_results[model].get(m, 0) for m in metrics[:4]],
                marker_color=colors[i % len(colors)],
                text=[f"{baseline_results[model].get(m, 0):.4f}" for m in metrics[:4]],
                textposition="outside",
                textfont=dict(size=11),
            ))

        fig_base.update_layout(
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_base, use_container_width=True)

    st.markdown("---")

    # --- Ablation results ---
    if ablation_results:
        st.markdown("### Ablation Study (V1 / V2 / V3)")
        st.caption("Evaluated on negative-sampled candidate pool — internal comparison")

        variants = list(ablation_results.keys())
        metrics_ab = ["Hit@10", "NDCG@10", "Recall@10", "MRR"]

        fig_ab = go.Figure()
        ab_colors = {"V1 (CF)": "#64748b", "V2 (CF+Content)": "#a78bfa",
                     "V3 (CF+Content+KG)": "#22d3ee"}
        for variant in variants:
            fig_ab.add_trace(go.Bar(
                name=variant,
                x=metrics_ab,
                y=[ablation_results[variant].get(m, 0) for m in metrics_ab],
                marker_color=ab_colors.get(variant, "#6366f1"),
                text=[f"{ablation_results[variant].get(m, 0):.4f}" for m in metrics_ab],
                textposition="outside",
                textfont=dict(size=11),
            ))

        fig_ab.update_layout(
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown("---")

    # --- Feature Importance ---
    if feat_importance and "V3 (CF+Content+KG)" in feat_importance:
        st.markdown("### Feature Importance (V3 — KG-Enhanced)")

        imp = feat_importance["V3 (CF+Content+KG)"]
        imp_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in imp.items()
        ]).sort_values("Importance", ascending=True)

        # Color KG features differently
        imp_df["Type"] = imp_df["Feature"].apply(
            lambda x: "KG Feature" if x.startswith("kg_") else "Base Feature"
        )

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Type",
            color_discrete_map={"KG Feature": "#22d3ee", "Base Feature": "#a78bfa"},
            template="plotly_dark",
        )
        fig_imp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=max(300, len(imp_df) * 35),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # --- Per-user NDCG distribution ---
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
                opacity=0.7,
                line_color="#e2e8f0",
            ))

        fig_dist.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="NDCG@10", gridcolor="#2a2a4a"),
            showlegend=True,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
