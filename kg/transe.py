"""
Train TransE knowledge graph embeddings.

TransE models each relation as a translation in embedding space:
  h + r ≈ t  for a true triple (h, r, t)

Trained with margin-based ranking loss against corrupted triples.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TripleDataset(Dataset):
    def __init__(self, triples, n_entities):
        self.triples = triples  # (N, 3) array of int ids
        self.n_entities = n_entities

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        # Corrupt head or tail
        if torch.rand(1).item() < 0.5:
            h_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h_neg, r, t
        else:
            t_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h, r, t_neg


class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=128):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_emb.weight.data = nn.functional.normalize(
                self.relation_emb.weight.data, p=2, dim=1
            )

    def forward(self, h, r, t):
        """Compute TransE distance: ||h + r - t||"""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)

    def normalize_entities(self):
        """Project entity embeddings onto unit ball."""
        with torch.no_grad():
            norms = self.entity_emb.weight.data.norm(p=2, dim=1, keepdim=True)
            norms = norms.clamp(min=1.0)
            self.entity_emb.weight.data.div_(norms)


def train_transe(
    triples_path="data/kg/triples.csv",
    entity2id_path="data/kg/entity2id.csv",
    output_dir="data/kg",
    dim=128,
    margin=1.0,
    lr=0.01,
    epochs=200,
    batch_size=1024,
    seed=42,
):
    """Train TransE and save entity/relation embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    triples_df = pd.read_csv(triples_path)
    entity2id_df = pd.read_csv(entity2id_path)

    entity2id = dict(zip(entity2id_df["entity"], entity2id_df["entity_id"]))
    n_entities = len(entity2id)

    # Build relation2id
    relations = triples_df["relation"].unique().tolist()
    relation2id = {r: i for i, r in enumerate(relations)}
    n_relations = len(relation2id)

    print(f"TransE: {n_entities} entities, {n_relations} relations, {len(triples_df)} triples")
    print(f"  dim={dim}, margin={margin}, lr={lr}, epochs={epochs}")

    # Convert triples to integer ids
    triple_ids = []
    skipped = 0
    for _, row in triples_df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        if h in entity2id and t in entity2id:
            triple_ids.append([entity2id[h], relation2id[r], entity2id[t]])
        else:
            skipped += 1
    triple_ids = np.array(triple_ids, dtype=np.int64)
    if skipped > 0:
        print(f"  Skipped {skipped} triples with unknown entities")

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransE(n_entities, n_relations, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TripleDataset(triple_ids, n_entities)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for h, r, t, h_neg, r_neg, t_neg in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)
            h_neg, r_neg, t_neg = h_neg.to(device), r_neg.to(device), t_neg.to(device)

            d_pos = model(h, r, t)
            d_neg = model(h_neg, r_neg, t_neg)
            loss = torch.clamp(margin + d_pos - d_neg, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_entities()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}")

    # Save embeddings
    entity_emb = model.entity_emb.weight.data.cpu().numpy()
    relation_emb = model.relation_emb.weight.data.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "transe_entity_emb.npy"), entity_emb)
    np.save(os.path.join(output_dir, "transe_relation_emb.npy"), relation_emb)

    with open(os.path.join(output_dir, "transe_relation2id.json"), "w") as f:
        json.dump(relation2id, f)

    print(f"  Saved: entity_emb {entity_emb.shape}, relation_emb {relation_emb.shape}")
    return entity_emb, relation_emb, entity2id, relation2id


def main():
    train_transe()


if __name__ == "__main__":
    main()
