# 实验计划：知识图谱增强的推荐系统（Phase 1: 数据到KG实验）

> 基于文档《Do Knowledge Graphs and Retrieval-Augmented Generation Improve Recommendation Performance and Explanation Faithfulness?》
> RAG相关实验（Phase 4-6）暂缓，待本阶段完成后再启动。

---

## 一、实验目标（当前阶段）

回答 **RQ1** 和 **RQ2**：

| 编号 | 研究问题 | 对应假设 |
|------|--------|--------|
| RQ1 | 知识图谱增强是否在推荐性能上显著优于协同过滤与内容模型？ | H1: 加入KG特征后，NDCG@K和Recall@K显著提升 |
| RQ2 | 知识图谱是否对long-tail电影和兴趣集中特征用户更有帮助？ | H2: KG对长尾电影召回提升更明显 |

---

## 二、资源与约束

| 项目 | 配置 |
|------|------|
| 数据集 | MovieLens 1M（~3,900部电影，~6,040用户，~100万条评分） |
| 元数据 | TMDB API（通过IMDb ID映射，断点续传） |
| GPU | 8 × V100 (16GB/32GB) |
| KG策略 | 先用简单特征，后续可扩展 |

---

## 三、系统架构（当前阶段）

```
MovieLens 1M + TMDB 元数据
    ↓
[数据预处理] 清洗、对齐、划分
    ↓
[Stage 1] 召回模型 (Item-CF / MF / LightGCN)
    ↓
[KG构建] 三元组 → NetworkX 图 → 手工图特征
    ↓
[Stage 2] 排序器 (LightGBM) → Ablation (V1/V2/V3)
    ↓
评估 + 分析
```

---

## 四、详细执行计划

### Phase 0：环境与数据准备

#### 0.1 环境搭建

```bash
# 核心依赖
pip install torch torchvision  # PyTorch (V100 CUDA)
pip install torch-geometric     # LightGCN
pip install lightgbm scikit-learn pandas numpy
pip install networkx            # KG 存储与查询
pip install sentence-transformers  # content similarity
pip install requests tqdm       # TMDB API
pip install scipy               # 统计检验
pip install matplotlib seaborn  # 可视化
```

项目目录：
```
544_project/
├── data/
│   ├── raw/              # ML-1M 原始数据
│   ├── processed/        # 清洗后数据 (train/test/neg samples)
│   ├── tmdb/             # TMDB 元数据 (断点续传缓存)
│   └── kg/               # 知识图谱三元组
├── src/
│   ├── data_prep/        # 数据获取与预处理
│   │   ├── download_ml1m.py
│   │   ├── fetch_tmdb.py        # 含断点续传
│   │   ├── align_data.py
│   │   └── split_data.py
│   ├── models/           # 推荐模型
│   │   ├── item_cf.py
│   │   ├── matrix_factorization.py
│   │   └── lightgcn.py
│   ├── kg/               # KG 构建与特征
│   │   ├── build_kg.py
│   │   └── kg_features.py
│   ├── ranker/           # 排序器
│   │   └── ranker.py
│   └── evaluation/       # 评估
│       └── metrics.py
├── configs/              # 超参数配置
├── results/              # 实验结果
└── notebooks/            # 分析与可视化
```

#### 0.2 下载 MovieLens 1M

- [ ] 下载 ML-1M 数据集
- [ ] 解析 `ratings.dat`, `movies.dat`, `users.dat`
- [ ] 转为 CSV 格式，统一编码
- [ ] 输出：`data/raw/ratings.csv`, `data/raw/movies.csv`

#### 0.3 获取 TMDB 元数据

- [ ] 从 ML-1M 的 `movies.dat` 中提取电影名 + 年份
- [ ] 通过 ML-1M 自带的 IMDb ID（`links.dat` 中有）映射到 TMDB
- [ ] **断点续传机制**：
  - 每次请求结果写入 `data/tmdb/cache/{movie_id}.json`
  - 重跑时跳过已缓存的电影
  - 记录失败列表 `data/tmdb/failed_ids.txt`
- [ ] 提取字段：genre, actor (top-5), director, overview, keywords, release_year
- [ ] 输出：`data/tmdb/tmdb_metadata.csv`
- [ ] 记录覆盖率（预计 80-90% 能匹配到）

#### 0.4 数据对齐与清洗

- [ ] 以 MovieLens movie_id 为主键，合并 TMDB 元数据
- [ ] 处理匹配失败的电影：保留（无KG特征时填零）
- [ ] 评分数据清洗：去重
- [ ] 用户最少交互过滤：保留交互数 ≥ 20 的用户
- [ ] 输出：`data/processed/clean_ratings.csv`, `data/processed/movie_metadata.csv`

#### 0.5 训练/测试集划分

- [ ] **按时间戳切分**：每个用户按时间排序，最后20%交互作为测试集
- [ ] 负采样：每个正样本采 4 个负样本（从用户未交互的电影中随机采样）
- [ ] **所有模型共用同一份负采样结果**，保证公平对比
- [ ] 输出：`data/processed/train.csv`, `data/processed/test.csv`

**Phase 0 检查点：**
- `clean_ratings.csv` 中的用户数、电影数、交互数
- TMDB 覆盖率
- train/test 比例

---

### Phase 1：Baseline 推荐模型

#### 1.1 统一评估接口

先实现评估模块，所有模型复用：

```python
# metrics.py
def evaluate(predictions, ground_truth, k=10):
    """
    predictions: dict[user_id] -> list[(movie_id, score)] 按 score 降序
    ground_truth: dict[user_id] -> set(movie_id)
    返回: Hit@K, NDCG@K, Recall@K, MRR
    """
```

指标定义：
| 指标 | 说明 |
|------|------|
| Hit@K | 测试集中有命中的用户比例 |
| NDCG@K | 归一化折损累积增益，考虑排序位置 |
| Recall@K | Top-K中命中测试集的比例 |
| MRR | 第一个命中位置的倒数均值 |
| Coverage | 被推荐过的电影占总电影的比例 |

#### 1.2 Item-CF (Baseline A)

- [ ] 构建 item-item 相似度矩阵（cosine similarity on user interaction vectors）
- [ ] 对每个用户，基于历史交互物品的相似物品加权打分
- [ ] 输出：每用户 Top-100 候选 + 分数
- [ ] 保存：`results/cf_scores.csv` (user_id, movie_id, cf_score)

#### 1.3 Matrix Factorization (Baseline B)

- [ ] BPR-MF 实现（PyTorch）
- [ ] 超参数搜索：

| 超参数 | 搜索范围 |
|--------|---------|
| embedding_dim | {32, 64, 128} |
| learning_rate | {1e-3, 5e-4, 1e-4} |
| regularization | {1e-4, 1e-5} |
| epochs | early stopping, patience=5 |

- [ ] 输出：`results/mf_scores.csv`

#### 1.4 LightGCN (Baseline C)

- [ ] 基于 PyTorch Geometric 实现
- [ ] 超参数搜索：

| 超参数 | 搜索范围 |
|--------|---------|
| num_layers | {2, 3} |
| embedding_dim | {64, 128} |
| learning_rate | {1e-3, 5e-4} |

- [ ] 利用多卡加速训练（V100 × 8 可以并行搜索超参数）
- [ ] 输出：`results/lightgcn_scores.csv`

#### 1.5 Baseline 汇总

- [ ] 运行评估，填写结果表：

| 模型 | Hit@10 | NDCG@10 | Recall@10 | MRR | Coverage |
|------|--------|---------|-----------|-----|----------|
| Item-CF | | | | | |
| MF | | | | | |
| LightGCN | | | | | |

**Phase 1 检查点：** 指标应与已有文献的 ML-1M 结果大致一致（如 MF 的 NDCG@10 通常在 0.03-0.06 范围，LightGCN 略高）。如果差异过大，需回头检查数据划分和负采样。

---

### Phase 2：知识图谱构建与特征工程

#### 2.1 KG 三元组构建

从 TMDB 元数据提取三类关系（先做简单的）：

| 关系 | 示例 | 说明 |
|------|------|------|
| `has_genre` | (Toy Story, has_genre, Animation) | 电影-类型 |
| `acted_by` | (Toy Story, acted_by, Tom Hanks) | 电影-演员（取 top-5 演员） |
| `directed_by` | (Toy Story, directed_by, John Lasseter) | 电影-导演 |

**暂不加入 `collaborated_with`**，因为 actor-movie-actor 的二跳路径已隐式包含此信息。后续如需可扩展。

- [ ] 构建三元组文件 `data/kg/triples.csv` (head, relation, tail)
- [ ] 构建实体映射 `data/kg/entity2id.csv`
- [ ] 用 NetworkX 构建图，提供查询接口
- [ ] 统计输出：节点数、边数、各关系类型数量

#### 2.2 手工图特征（简单方案）

对每个 (user, candidate_movie) 对，基于用户历史交互电影计算KG特征：

```python
def compute_kg_features(user_history_movies, candidate_movie, kg_graph):
    features = {}
    for hist_movie in user_history_movies:
        features['shared_actor_count'] += count_shared_actors(hist_movie, candidate)
        features['same_director'] |= has_same_director(hist_movie, candidate)
        features['same_genre_count'] += count_shared_genres(hist_movie, candidate)
        features['shortest_path_len'] = min_shortest_path(hist_movie, candidate)
    # 对多部历史电影取聚合 (平均/最大)
    return aggregate(features)
```

| 特征 | 类型 | 聚合方式 |
|------|------|---------|
| `shared_actor_count` | int | sum / mean |
| `same_director` | binary | max (有一部同导演即为1) |
| `same_genre_count` | int | sum / mean |
| `shortest_path_len` | int | min |

- [ ] 为 train + test 中所有 (user, candidate) 对计算特征
- [ ] 注意：shortest_path 计算较慢，对 ML-1M 规模可能需要优化（设最大路径长度限制，如 ≤ 4）
- [ ] 输出：`data/kg/kg_features.csv` (user_id, movie_id, shared_actor_count, same_director, same_genre_count, shortest_path_len)

#### 2.3 Content Similarity 特征

- [ ] 将电影元数据（genre + overview）拼接为文本
- [ ] 使用 Sentence-Transformer 编码为向量
- [ ] 计算 (user_history_movies, candidate) 的平均 cosine similarity
- [ ] 输出：加入到特征表中 `content_similarity` 列

#### 2.4 KG 分析（可视化）

- [ ] KG 图规模统计
- [ ] 各关系类型的分布
- [ ] 示例 KG 子图可视化（选几部电影画子图）

**Phase 2 检查点：** KG 特征覆盖率（多少 user-movie 对有非零KG特征），如果大量为零说明 TMDB 数据覆盖不足。

---

### Phase 3：排序器 + Ablation 实验

#### 3.1 排序器训练

使用 LightGBM，pointwise 方式（label = 1/0）：

| Variant | 输入特征 | 目的 |
|---------|---------|------|
| **V1** | cf_score | 纯协同过滤 |
| **V2** | cf_score + content_similarity + popularity | 加入内容特征 |
| **V3** | cf_score + content_similarity + popularity + KG_features | 加入KG特征 |

其中 `cf_score` 取三个 baseline 中最好的那个（或分别跑三次）。

**控制变量：**
- 三个 Variant 使用完全相同的 train/test 划分
- 相同的候选集（相同的负采样）
- LightGBM 超参数统一搜索

```python
lgbm_params = {
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_samples': [20, 50],
}
```

#### 3.2 实验一结果表（回答 RQ1）

- [ ] 填写 Ablation 结果：

| Variant | Hit@10 | NDCG@10 | Recall@10 | MRR | Coverage |
|---------|--------|---------|-----------|-----|----------|
| V1 (CF) | | | | | |
| V2 (CF+Content) | | | | | |
| V3 (CF+Content+KG) | | | | | |

- [ ] **统计显著性检验**：对 V2 vs V3，用每个用户的 NDCG@10 做 paired t-test

```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(v3_ndcg_per_user, v2_ndcg_per_user)
# p < 0.05 → 差异显著
```

- [ ] 绘制 Ablation 柱状图

#### 3.3 Long-tail 分析（回答 RQ2）

- [ ] 定义 long-tail：按电影出现次数排序，bottom 50% 为 tail
- [ ] 分别计算 V1/V2/V3 在 head 和 tail 电影上的 Recall@10

| Variant | Head Recall@10 | Tail Recall@10 | Tail提升幅度 |
|---------|---------------|----------------|------------|
| V1 | | | — |
| V2 | | | |
| V3 | | | |

- [ ] 绘制 popularity vs recall 曲线
- [ ] 用户兴趣熵分析（可选）：按 genre entropy 分桶，看 KG 对哪类用户帮助更大

#### 3.4 特征重要性分析

- [ ] 从 LightGBM 获取 feature importance (gain / split)
- [ ] 绘制 KG 特征贡献图
- [ ] 分析哪些 KG 特征最有价值

**Phase 3 检查点：**
- V3 在至少 NDCG@10 或 Recall@10 上优于 V2
- 如果 KG 没有提升，分析原因：特征质量？数据稀疏？需要更丰富的关系？
- 如果 KG 特征的 feature importance 很低，考虑扩展特征（加入 `collaborated_with`、KG embedding 等）

---

## 五、时间线

```
Week 1:
  ├── [Phase 0] 环境搭建 + 下载 ML-1M + TMDB 数据获取（断点续传）
  └── [Phase 0] 数据对齐、清洗、划分

Week 2:
  ├── [Phase 1] 实现 Item-CF + MF + LightGCN
  ├── [Phase 1] 超参数调优（利用多卡并行）
  └── [Phase 1] Baseline 评估

Week 3:
  ├── [Phase 2] KG 三元组构建
  ├── [Phase 2] 手工图特征 + Content Similarity
  └── [Phase 2] KG 分析与可视化

Week 4:
  ├── [Phase 3] LightGBM 排序器（V1/V2/V3）
  ├── [Phase 3] Ablation 实验 + 统计检验
  ├── [Phase 3] Long-tail 分析
  └── [Phase 3] 特征重要性分析 + 结论整理
```

---

## 六、后续扩展（本阶段完成后）

完成 Phase 0-3 后，根据结果决定下一步：
1. **如果 KG 有效** → 进入 RAG 解释模块（Phase 4-6）
2. **如果 KG 提升不显著** → 先扩展 KG 特征（加 `collaborated_with`、KG embedding），再决定是否继续 RAG
3. **RAG 实验** → 构建文本证据库、实现检索、Prompt 设计、Faithfulness 评估

---

*计划生成时间: 2026-03-19*
*当前阶段: Phase 0 → Phase 3（数据到KG实验）*
