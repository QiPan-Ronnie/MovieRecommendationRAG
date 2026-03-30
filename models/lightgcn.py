"""
LightGCN implementation using PyTorch.
Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.
Supports validation-based early stopping.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.sparse import coo_matrix

from evaluation.metrics import evaluate_all, print_results


class BPRSampler(Dataset):
    """BPR training sampler."""

    def __init__(self, train_df, n_items, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_items = n_items
        self.interactions = list(
            zip(train_df["user_idx"].values, train_df["movie_idx"].values)
        )
        self.user_pos = defaultdict(set)
        for u, i in self.interactions:
            self.user_pos[u].add(i)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, pos = self.interactions[idx]
        neg = self.rng.randint(0, self.n_items)
        while neg in self.user_pos[u]:
            neg = self.rng.randint(0, self.n_items)
        return u, pos, neg


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

        self.adj = None

    def compute_embeddings(self):
        """LightGCN forward: average neighbor embeddings across layers."""
        all_embed = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        embeddings_list = [all_embed]
        for _ in range(self.n_layers):
            all_embed = torch.sparse.mm(self.adj, all_embed)
            embeddings_list.append(all_embed)

        final_embed = torch.stack(embeddings_list, dim=0).mean(dim=0)
        user_emb = final_embed[:self.n_users]
        item_emb = final_embed[self.n_users:]
        return user_emb, item_emb

    def forward(self, users, pos_items, neg_items):
        user_emb, item_emb = self.compute_embeddings()

        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]

        pos_score = (u * p).sum(dim=1)
        neg_score = (u * n).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        reg_loss = (
            self.user_embed(users).norm(2).pow(2) +
            self.item_embed(pos_items).norm(2).pow(2) +
            self.item_embed(neg_items).norm(2).pow(2)
        ) / len(users)

        return loss, reg_loss

    def predict_all(self, user_indices):
        """Predict scores for all items for given users."""
        user_emb, item_emb = self.compute_embeddings()
        u = user_emb[user_indices]
        scores = torch.matmul(u, item_emb.T)
        return scores


def build_adj_matrix(train_df, n_users, n_items, device):
    """Build normalized adjacency matrix for LightGCN."""
    users = train_df["user_idx"].values
    items = train_df["movie_idx"].values + n_users

    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    vals = np.ones(len(rows))

    n_nodes = n_users + n_items
    adj = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    degree = np.array(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    row_norm = degree_inv_sqrt[rows]
    col_norm = degree_inv_sqrt[cols]
    normalized_vals = row_norm * vals * col_norm

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values = torch.tensor(normalized_vals, dtype=torch.float32)
    adj_tensor = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).to(device)
    return adj_tensor


def evaluate_on_split(model, split_df, train_user_items, user2idx, movie2idx,
                      idx2movie, device, k=10):
    """Evaluate model on a given split."""
    ground_truth = defaultdict(set)
    for _, row in split_df.iterrows():
        ground_truth[row["user_id"]].add(row["movie_id"])

    model.eval()
    predictions = {}
    split_users = [u for u in ground_truth.keys() if u in user2idx]

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(split_users), batch_size):
            batch_users = split_users[start:start + batch_size]
            user_indices = torch.tensor([user2idx[u] for u in batch_users]).to(device)
            scores = model.predict_all(user_indices).cpu().numpy()

            for i, uid in enumerate(batch_users):
                user_scores = scores[i]
                for mid in train_user_items[uid]:
                    if mid in movie2idx:
                        user_scores[movie2idx[mid]] = -float("inf")

                top_indices = np.argsort(user_scores)[::-1][:100]
                predictions[uid] = [idx2movie[idx] for idx in top_indices]

    total_items = len(movie2idx)
    results, per_user_ndcg = evaluate_all(predictions, ground_truth, k=k, total_items=total_items, ks=[1, 5, 10])
    return results, per_user_ndcg, predictions


def train_lightgcn(train_df, embed_dim=64, n_layers=3, lr=1e-3, reg=1e-5,
                   epochs=100, batch_size=2048, patience=10, device="cpu",
                   val_df=None):
    """Train LightGCN model with optional validation-based early stopping."""
    all_users = sorted(train_df["user_id"].unique())
    all_movies = sorted(train_df["movie_id"].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    movie2idx = {m: i for i, m in enumerate(all_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    n_users = len(all_users)
    n_items = len(all_movies)

    train_indexed = train_df.copy()
    train_indexed["user_idx"] = train_indexed["user_id"].map(user2idx)
    train_indexed["movie_idx"] = train_indexed["movie_id"].map(movie2idx)

    adj = build_adj_matrix(train_indexed, n_users, n_items, device)

    model = LightGCN(n_users, n_items, embed_dim, n_layers).to(device)
    model.adj = adj

    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = BPRSampler(train_indexed, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Build train user items for masking
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["movie_id"])

    best_val_ndcg = -1
    best_loss = float("inf")
    val_no_improve = 0
    loss_no_improve = 0
    best_state = None
    use_val = val_df is not None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for users, pos_items, neg_items in loader:
            users = users.long().to(device)
            pos_items = pos_items.long().to(device)
            neg_items = neg_items.long().to(device)

            bpr_loss, reg_loss = model(users, pos_items, neg_items)
            loss = bpr_loss + reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if use_val and (epoch + 1) % 10 == 0:
            # Validation-based early stopping (checked every 10 epochs)
            val_results, _, _ = evaluate_on_split(
                model, val_df, train_user_items, user2idx, movie2idx,
                idx2movie, device, k=10
            )
            val_ndcg = val_results["NDCG@10"]
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val NDCG@10: {val_ndcg:.4f}")
            if val_ndcg > best_val_ndcg + 1e-4:
                best_val_ndcg = val_ndcg
                val_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                val_no_improve += 1
                if val_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (val NDCG)")
                    break
        elif not use_val:
            # Loss-based early stopping (only when no validation set)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                loss_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                loss_no_improve += 1
                if loss_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (train loss)")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, user2idx, movie2idx, idx2movie


def run_lightgcn(train_path="data/processed/train.csv",
                 val_path="data/processed/val.csv",
                 test_path="data/processed/test.csv",
                 output_path="results/lightgcn_scores.csv",
                 embed_dim=64, n_layers=3, lr=1e-3, epochs=100, k=10):
    """Run full LightGCN pipeline with validation."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Training LightGCN (dim={embed_dim}, layers={n_layers})...")
    model, user2idx, movie2idx, idx2movie = train_lightgcn(
        train_df, embed_dim=embed_dim, n_layers=n_layers, lr=lr,
        epochs=epochs, device=device, val_df=val_df
    )

    # Build train user items
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row["user_id"]].add(row["movie_id"])

    # Evaluate on test set
    print("Evaluating on test set...")
    results, per_user_ndcg, predictions = evaluate_on_split(
        model, test_df, train_user_items, user2idx, movie2idx,
        idx2movie, device, k=k
    )

    # Save scores
    all_scores = []
    split_users = [u for u in predictions.keys() if u in user2idx]
    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(split_users), batch_size):
            batch_users = split_users[start:start + batch_size]
            user_indices = torch.tensor([user2idx[u] for u in batch_users]).to(device)
            scores = model.predict_all(user_indices).cpu().numpy()

            for i, uid in enumerate(batch_users):
                for movie_id in predictions[uid]:
                    if movie_id in movie2idx:
                        all_scores.append({
                            "user_id": uid,
                            "movie_id": movie_id,
                            "lgcn_score": float(scores[i][movie2idx[movie_id]])
                        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(all_scores).to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

    print_results(results, f"LightGCN (dim={embed_dim}, layers={n_layers})")
    return results, per_user_ndcg


if __name__ == "__main__":
    run_lightgcn()
