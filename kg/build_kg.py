"""
Build Knowledge Graph from TMDB metadata + collaborative edges.

Relations:
  - has_genre, acted_by, directed_by (from TMDB metadata)
  - released_in_decade (from TMDB year)
  - co_liked (from user co-interaction in training data, threshold ≥ N users)
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from collections import Counter, defaultdict


def load_tmdb_metadata(path="data/tmdb/tmdb_metadata.csv"):
    """Load TMDB metadata."""
    df = pd.read_csv(path)
    return df


def build_triples(metadata_df):
    """
    Build KG triples from TMDB metadata.
    Returns list of (head, relation, tail) tuples.
    """
    triples = []
    entity_types = {}  # entity -> type

    for _, row in metadata_df.iterrows():
        movie_id = int(row["movie_id"])
        movie_node = f"movie_{movie_id}"
        entity_types[movie_node] = "movie"

        # Genre relations
        genres = str(row.get("genres", ""))
        if genres and genres != "nan":
            for genre in genres.split("|"):
                genre = genre.strip()
                if genre:
                    genre_node = f"genre_{genre}"
                    entity_types[genre_node] = "genre"
                    triples.append((movie_node, "has_genre", genre_node))

        # Actor relations
        actors = str(row.get("actors", ""))
        if actors and actors != "nan":
            for actor in actors.split("|"):
                actor = actor.strip()
                if actor:
                    actor_node = f"actor_{actor}"
                    entity_types[actor_node] = "actor"
                    triples.append((movie_node, "acted_by", actor_node))

        # Director relations
        directors = str(row.get("directors", ""))
        if directors and directors != "nan":
            for director in directors.split("|"):
                director = director.strip()
                if director:
                    director_node = f"director_{director}"
                    entity_types[director_node] = "director"
                    triples.append((movie_node, "directed_by", director_node))

    return triples, entity_types


def build_decade_triples(metadata_df):
    """Add released_in_decade relations."""
    triples = []
    entity_types = {}
    for _, row in metadata_df.iterrows():
        movie_id = int(row["movie_id"])
        movie_node = f"movie_{movie_id}"
        year = row.get("year")
        if pd.notna(year):
            decade = int(int(year) // 10) * 10
            decade_node = f"decade_{decade}s"
            entity_types[decade_node] = "decade"
            triples.append((movie_node, "released_in_decade", decade_node))
    return triples, entity_types


def build_collaborative_triples(
    train_path="data/processed/train.csv",
    min_cooccurrence=10,
    max_edges=100000,
):
    """
    Add co_liked edges between movies frequently co-liked by users.

    This encodes collaborative filtering signal directly into the KG,
    bridging the gap between CF and KG-based recommendations.
    """
    train_df = pd.read_csv(train_path)
    user_movies = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Count co-occurrences
    cooccur = Counter()
    for uid, movies in user_movies.items():
        movies = sorted(movies)
        for i in range(len(movies)):
            for j in range(i + 1, len(movies)):
                cooccur[(movies[i], movies[j])] += 1

    # Filter by threshold and sort by count
    strong_pairs = [(pair, count) for pair, count in cooccur.items()
                    if count >= min_cooccurrence]
    strong_pairs.sort(key=lambda x: x[1], reverse=True)
    strong_pairs = strong_pairs[:max_edges]

    triples = []
    for (m1, m2), count in strong_pairs:
        node1 = f"movie_{m1}"
        node2 = f"movie_{m2}"
        triples.append((node1, "co_liked", node2))

    print(f"  Collaborative edges: {len(triples)} pairs "
          f"(threshold >= {min_cooccurrence} users, "
          f"from {len(cooccur)} total pairs)")
    return triples


def build_networkx_graph(triples):
    """Build a NetworkX graph from triples."""
    G = nx.Graph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G


def print_kg_stats(triples, entity_types, G):
    """Print KG statistics."""
    relation_counts = Counter(r for _, r, _ in triples)
    type_counts = Counter(entity_types.values())

    print("\n" + "=" * 50)
    print("  Knowledge Graph Statistics")
    print("=" * 50)
    print(f"  Total triples: {len(triples)}")
    print(f"  Total entities: {len(entity_types)}")
    print(f"  Graph nodes: {G.number_of_nodes()}")
    print(f"  Graph edges: {G.number_of_edges()}")
    print(f"\n  Entity types:")
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype}: {count}")
    print(f"\n  Relation types:")
    for rel, count in sorted(relation_counts.items()):
        print(f"    {rel}: {count}")
    print("=" * 50)


def main():
    out_dir = "data/kg"
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata
    metadata = load_tmdb_metadata()
    print(f"Loaded metadata for {len(metadata)} movies")

    # Build metadata triples
    triples, entity_types = build_triples(metadata)
    print(f"  Metadata triples: {len(triples)}")

    # Add decade relations
    decade_triples, decade_entities = build_decade_triples(metadata)
    triples.extend(decade_triples)
    entity_types.update(decade_entities)
    print(f"  + Decade triples: {len(decade_triples)}")

    # Add collaborative edges
    collab_triples = build_collaborative_triples()
    triples.extend(collab_triples)

    # Save triples
    triples_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    triples_df.to_csv(os.path.join(out_dir, "triples.csv"), index=False)

    # Build entity mapping
    entities = sorted(entity_types.keys())
    entity2id = {e: i for i, e in enumerate(entities)}
    entity_df = pd.DataFrame([
        {"entity": e, "entity_id": entity2id[e], "type": entity_types[e]}
        for e in entities
    ])
    entity_df.to_csv(os.path.join(out_dir, "entity2id.csv"), index=False)

    # Build NetworkX graph
    G = build_networkx_graph(triples)

    # Save graph
    with open(os.path.join(out_dir, "kg_graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    # Stats
    print_kg_stats(triples, entity_types, G)

    print(f"\nFiles saved to {out_dir}/")
    print(f"  - triples.csv ({len(triples)} triples)")
    print(f"  - entity2id.csv ({len(entities)} entities)")
    print(f"  - kg_graph.pkl (NetworkX graph)")


if __name__ == "__main__":
    main()
