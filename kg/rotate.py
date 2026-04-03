"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

Models each relation as a rotation in complex space:
  t = h ∘ r  (element-wise complex multiplication)

Advantages over TransE:
  - Handles symmetric relations (co_liked): rotation by π
  - Handles 1-to-N / N-to-N relations (has_genre, acted_by)
  - Better separation in embedding space
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
        self.triples = triples
        self.n_entities = n_entities

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        if torch.rand(1).item() < 0.5:
            h_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h_neg, r, t
        else:
            t_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h, r, t_neg


class BalancedTripleDataset(Dataset):
    """Relation-balanced sampling."""
    def __init__(self, triples, n_entities):
        self.n_entities = n_entities
        self.relation_groups = {}
        for i in range(len(triples)):
            r = int(triples[i, 1])
            if r not in self.relation_groups:
                self.relation_groups[r] = []
            self.relation_groups[r].append(i)
        self.relations = sorted(self.relation_groups.keys())
        self.n_relations = len(self.relations)
        self.max_group_size = max(len(v) for v in self.relation_groups.values())
        self.triples = triples

    def __len__(self):
        return self.max_group_size * self.n_relations

    def __getitem__(self, idx):
        rel_idx = idx % self.n_relations
        r_type = self.relations[rel_idx]
        group = self.relation_groups[r_type]
        sample_idx = group[torch.randint(0, len(group), (1,)).item()]
        h, r, t = self.triples[sample_idx]
        if torch.rand(1).item() < 0.5:
            h_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h_neg, r, t
        else:
            t_neg = torch.randint(0, self.n_entities, (1,)).item()
            return h, r, t, h, r, t_neg


class RotatE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=128, gamma=12.0):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.epsilon = 2.0

        # Entity embeddings in complex space (dim = 2 * half_dim)
        self.half_dim = dim // 2
        self.entity_emb = nn.Embedding(n_entities, dim)
        # Relation embeddings: phase angles (half_dim)
        self.relation_emb = nn.Embedding(n_relations, self.half_dim)

        embedding_range = (self.gamma.item() + self.epsilon) / self.half_dim
        nn.init.uniform_(self.entity_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.relation_emb.weight, -np.pi, np.pi)

    def forward(self, h_idx, r_idx, t_idx):
        """Compute RotatE distance: ||h ∘ r - t||"""
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        r_phase = self.relation_emb(r_idx)

        # Split into real and imaginary parts
        h_re, h_im = h[..., :self.half_dim], h[..., self.half_dim:]
        t_re, t_im = t[..., :self.half_dim], t[..., self.half_dim:]

        # Relation as rotation: r = (cos(θ), sin(θ))
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # Complex multiplication: h ∘ r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Distance
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        score = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-9).sum(dim=-1)
        return score


def train_rotate(
    triples_path="data/kg/triples.csv",
    entity2id_path="data/kg/entity2id.csv",
    output_dir="data/kg",
    dim=128,
    gamma=12.0,
    lr=0.001,
    epochs=300,
    batch_size=1024,
    seed=42,
    balanced=False,
    negative_sample_size=1,
):
    """Train RotatE and save entity embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    triples_df = pd.read_csv(triples_path)
    entity2id_df = pd.read_csv(entity2id_path)
    entity2id = dict(zip(entity2id_df["entity"], entity2id_df["entity_id"]))
    n_entities = len(entity2id)

    relations = triples_df["relation"].unique().tolist()
    relation2id = {r: i for i, r in enumerate(relations)}
    n_relations = len(relation2id)

    print(f"RotatE: {n_entities} entities, {n_relations} relations, {len(triples_df)} triples")
    print(f"  dim={dim}, gamma={gamma}, lr={lr}, epochs={epochs}")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = RotatE(n_entities, n_relations, dim, gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if balanced:
        print(f"  Using relation-balanced sampling")
        dataset = BalancedTripleDataset(triple_ids, n_entities)
    else:
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
            # Self-adversarial negative sampling loss
            loss = torch.clamp(model.gamma + d_pos - d_neg, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}")

    # Save embeddings
    entity_emb = model.entity_emb.weight.data.cpu().numpy()
    relation_phase = model.relation_emb.weight.data.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "transe_entity_emb.npy"), entity_emb)
    np.save(os.path.join(output_dir, "transe_relation_emb.npy"), relation_phase)

    with open(os.path.join(output_dir, "transe_relation2id.json"), "w") as f:
        json.dump(relation2id, f)

    print(f"  Saved: entity_emb {entity_emb.shape}, relation_phase {relation_phase.shape}")
    return entity_emb, relation_phase, entity2id, relation2id


def main():
    train_rotate()


if __name__ == "__main__":
    main()
