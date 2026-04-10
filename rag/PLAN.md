# RAG 模块落地计划

## 已完成

### A. 代码骨架搭建 + Bug 修复 ✅

| 文件 | 内容 |
|------|------|
| `rag/__init__.py` | 模块入口 |
| `rag/build_corpus.py` | TMDB 元数据 -> 句级 chunk -> FAISS + BM25 索引 |
| `rag/retriever.py` | `HybridRetriever` 类：双路检索 score = α·dense + (1-α)·BM25 |
| `rag/generator.py` | 结构化 Prompt 设计（RAG / prompt-only），支持 HuggingFace 本地 + API 后端 |
| `rag/faithfulness.py` | 4 种指标（Overlap / ROUGE-L / Semantic Sim / BERTScore）+ E1-E4 扰动函数 |
| `rag/pipeline.py` | 主编排器 Phase 5.1-5.4，命令行参数控制，已集成 `run_all.py` |
| `rag/test_retrieval.py` | 检索质量测试脚本 |

**修复的问题**：
- `pipeline.py` 中 `get_top_k_recommendations()` 改为从 `multi_recall_scores.csv` 加载
- 添加了 `_load_existing_results()` / `_append_result()` 实现**断点续传**
- 修复了 Windows 环境下 Unicode 箭头打印报错
- 已集成到 `run_all.py` 作为 `--phase 5`

### B. 证据语料库构建 ✅

已运行 `python -m rag.pipeline --phase 5.1`：
- **18,878 个文档 chunk**（来自 3,652 部电影）
- 来源分布：overview（剧情句）、tagline、genre 描述、cast 描述
- 输出文件在 `data/rag/`：corpus.jsonl, faiss_index.bin, corpus_embeddings.npy, bm25_index.pkl, doc_id_to_movie.json

### C. 检索质量验证 ✅

已运行 `python -m rag.test_retrieval` 验证：
- 候选电影的 genre / overview / cast 证据能被正确检索到
- 用户历史电影的信息也会被包含（解释"为什么推荐"）
- dense + sparse 混合检索效果好于单一路径
- alpha=0.6 是合理的默认值

### D. 环境搭建 ✅

- Conda 环境 `CSCI544`（Python 3.11）
- PyTorch nightly + CUDA 12.8（支持 RTX 5070 Ti Blackwell 架构）
- 所有依赖已安装：faiss-cpu, rank-bm25, sentence-transformers, transformers, bert-score 等

---

## 待完成（AutoDL 服务器上）

### D. LLM 生成解释（Phase 5.2）🔲

**目标**：用 Qwen2.5-7B-Instruct 生成 RAG + prompt-only 两种模式的解释

**步骤**：
1. 在 AutoDL 上配置环境（安装依赖，上传代码和 `data/rag/`）
2. 小规模测试：
   ```bash
   python -m rag.pipeline --phase 5.2 --num-users 10
   ```
3. 中规模验证（500 用户 = 5000 条）确认质量
4. 全量运行（5950 用户 × top-10 = ~59500 条，预计 8-16 小时）
5. 输出：`results/explanations_rag.jsonl` + `results/explanations_prompt_only.jsonl`

**注意事项**：
- 7B 模型需要 ~14GB VRAM（FP16）或 ~7GB（4bit 量化）
- 断点续传已实现，中断后重新运行会跳过已生成的记录
- 如用 vLLM 部署可切换 `--llm-backend api --api-url http://localhost:8000/v1`

### D.4 KG-only 解释模式（RQ4 对比用）🔲

在 `generator.py` 中添加第三种模式：
- 用 KG 特征（共享演员、导演、类型）作为结构化证据
- 从 `data/kg/kg_features_*.csv` 提取路径信息
- 与 RAG-only、KG+RAG 对比（RQ4 的核心实验）

### E. 扰动实验 E1-E4（Phase 5.3）🔲

```bash
python -m rag.pipeline --phase 5.3 --num-samples 200
```
- 200 样本 × 4 条件 = 800 次 LLM 生成
- E1（原始）→ E2（删关键证据）→ E3（乱序）→ E4（替换无关证据）
- 预期结果：E1 最好，E4 最差 → 证明模型确实使用了检索证据

**待优化**：
- `perturb_e4_replace_unrelated` 加 genre 过滤，确保替换的电影与候选类型差异大
- `perturb_e2_remove_key` 确保移除后至少保留 2-3 条证据

### F. 忠实度评估 & 可视化（Phase 5.4）🔲

```bash
python -m rag.pipeline --phase 5.4
```

**需新建 `rag/visualize.py`**：
- 图 1：RAG vs Prompt-only 指标柱状图
- 图 2：E1-E4 扰动实验折线图
- 图 3：KG-only vs RAG-only vs KG+RAG 对比（RQ4）
- 图 4：3-5 个 case study 案例展示
- 统计显著性：RAG vs prompt-only 配对 t 检验，E1 vs E4 配对 t 检验

### G. Streamlit Demo 集成 🔲

在现有 `app.py` 中添加 RAG Tab：
- 预加载 `explanations_rag.jsonl`（不需要实时生成）
- 展示：推荐列表 + KG 图 + 检索证据 + RAG 解释
- 对比展示 prompt-only vs RAG 解释质量差异

---

## 命令速查

```bash
# 在 AutoDL 上
conda activate CSCI544  # 或对应环境
cd MovieRecommendation

# 构建语料库（已完成，如需重建）
python -m rag.pipeline --phase 5.1

# 小规模测试生成
python -m rag.pipeline --phase 5.2 --num-users 10

# 全量生成
python -m rag.pipeline --phase 5.2

# 扰动实验
python -m rag.pipeline --phase 5.3 --num-samples 200

# 评估
python -m rag.pipeline --phase 5.4

# 一键全跑
python run_all.py --phase 5
```

---

## 文件结构总览

```
rag/
├── __init__.py           # 模块入口
├── build_corpus.py       # Phase 5.1: TMDB -> chunks -> FAISS + BM25
├── retriever.py          # 混合检索器 (HybridRetriever)
├── generator.py          # LLM 解释生成 (RAG / prompt-only)
├── faithfulness.py       # 忠实度指标 + E1-E4 扰动
├── pipeline.py           # 主编排器 (Phase 5.1-5.4)
├── test_retrieval.py     # 检索质量测试
├── visualize.py          # [待建] 结果可视化
└── PLAN.md               # 本文件

data/rag/                 # 已生成
├── corpus.jsonl          # 18,878 docs
├── faiss_index.bin       # FAISS 向量索引
├── corpus_embeddings.npy # 嵌入矩阵 (18878 × 384)
├── bm25_index.pkl        # BM25 稀疏索引
└── doc_id_to_movie.json  # doc_id -> movie_id 映射
```
