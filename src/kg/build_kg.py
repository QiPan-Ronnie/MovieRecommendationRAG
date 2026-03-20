"""
Build Knowledge Graph from TMDB metadata.
Constructs triples: (movie, has_genre, genre), (movie, acted_by, actor), (movie, directed_by, director)
"""
import os
import pandas as pd
import networkx as nx
import pickle
from collections import Counter


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

    # Build triples
    triples, entity_types = build_triples(metadata)

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
