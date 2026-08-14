# cellfish 实施计划

仓库：[https://github.com/xfchen0912/cellfish](https://github.com/xfchen0912/cellfish)

从 MASLD-HCC 仓库的 `src/sc_helpers` 抽出一套个人常用、可跨课题复用的单细胞工具包。工作名 **`cellfish`**。`sc_helpers` 先当来源，不一次性搬家；本仓库当前是阶段 0–1 骨架。

## 目标

- 现在：scRNA + scATAC / multiome
- 以后：空间（Visium / 成像空间），不预建空的 `spatial/` 包
- 模块浅：层 1 公用原语 + `ext/<算法>/` + 课题配置留在分析仓库
- 少造轮子：能直接用 scanpy / muon / pandas / matplotlib 的不包

## 非目标

- 不做 QC / 整合 / peak calling / spaceranger 流水线
- 不包模型训练（ChromBPNet 训练等留在 notebook / shell）
- 不引入插件系统、ABC、深目录
- 不把 MASLD-HCC 的路径、色板、`bulk_helpers`、Fig 专用图放进包
- 第一期不发布正式 PyPI；结构按「以后能 `pip install`」来

## 三层边界

| 层 | 放什么 | 换课题后 |
| --- | --- | --- |
| **1 公用** | 与算法无关的 io / 检查 / 配对 / 字体色板 / 通用图 | 仍会用 |
| **2 `ext/<tool>/`** | 一个算法一个文件夹，计算和出图在一起 | 不装该工具就不需要 |
| **3 课题** | `DATA_DIR`、论文色板、队列 registry、Fig 函数 | 必须改 |

**依赖方向：** `ext` → 层 1；层 1 禁止 import `ext`。

**判断标准：** 换课题、换物种、不装 DRVI / ChromBPNet 也会用 → 层 1；离开该算法就没有意义 → `ext/<tool>/`；换仓库路径或论文配色就会变 → 层 3。

**空间：** 第三种坐标（`obsm["spatial"]` + 可选底图），不是第三套顶层包。绘图统一 `basis=`。

不要按 RNA / ATAC / 空间切顶层目录。Visium 出图和 UMAP 出图大部分是「2D 坐标 + obs 着色」，差别只是坐标从哪来、要不要垫一张组织图。

## 三种坐标系（不是三种模块）

| 空间 | 来源 | 底图 |
| --- | --- | --- |
| 细胞嵌入 | `adata.obsm["X_umap"]` | 无 |
| 组织坐标 | `adata.obsm["spatial"]` | 可选 H&E / IF |
| 基因组座 | chrom, start, end | 可选 bigWig / contrib |

函数用 `basis=` / `coords=` 切换，不要 `plot_umap` / `plot_spatial` / `plot_peak` 三套平行 API。`var` 不默认是基因（可以是 peak / cCRE / motif / bin）。

对象模型：现在 AnnData + MuData；空间未到不要上 SpatialData。不要自造第四种对象。

---

## 目标目录

```text
cellfish/
├── __init__.py                 # 只暴露 io / data / pl / stats / ext
├── io/
│   └── _anndata.py             # sanitize + write_h5_safe
├── data/
│   ├── _check.py               # 新增：obs/obsm/var/layer
│   ├── _operations.py          # marker、前后缀（已有）
│   └── _pairing.py             # 从 plot 挪出的 join
├── plot/
│   ├── _fonts.py
│   ├── _style.py               # 新增：setup_style / savefig
│   ├── _palettes.py            # 只留通用色表和函数
│   ├── _embedding.py           # 合并 _single + _scatterplot 的对外 API
│   ├── _dotplot.py / _grid_dotplot.py
│   ├── _proportions.py / _ridgeplot.py
│   └── _plot1cell.py / _plot1cell_atlas.py
├── stats/
│   └── _stats.py               # 保持极薄
└── ext/
    ├── __init__.py             # __getattr__ 惰性加载
    ├── drvi/                   # 现 utils/_drvi_utils.py
    ├── milo/                   # 现 plot/_milo.py
    ├── scenicplus/
    ├── chrombpnet/             # contrib / locus / gene_ccre
    ├── liana/
    └── archr/                  # create_mudata_from_ArchProj
```

公开用法：

```python
import cellfish as cf

cf.pl.embedding(adata, color="cell_type", basis="X_umap")
cf.io.write_h5_safe(adata, path)
cf.data.require_obs(adata, ["cell_type"])
cf.ext.drvi.plot_latent_heatmap(...)
```

根 `__init__.py` 保持很瘦：

```python
from . import plot as pl
from . import io, data, stats
from . import ext

__all__ = ["pl", "io", "data", "stats", "ext"]
```

### `ext/<tool>/` 内部

每个算法文件夹默认两类文件：

```text
ext/drvi/
├── __init__.py     # 对外 API
├── _prep.py        # 数据变换
└── _plot.py        # 本算法的图（调 cf.pl 原语）
```

- 单文件不够大就先一个 `_core.py`
- 超过约 400–600 行，或依赖不同（h5 读取 vs matplotlib）再拆
- 不要为对称预留空的 `_enrich.py` / `_stats.py`
- 不要再建 `plot/_<tool>.py` + `utils/_<tool>_utils.py` 的分叉

`ext/__init__.py` 用惰性加载，避免 `import cellfish` 把 ChromBPNet 的 `h5py` / `logomaker` 全拉进来：

```python
from importlib import import_module

def __getattr__(name: str):
    return import_module(f"{__name__}.{name}")
```

算法文件夹可以依赖 `io/`、`data/`、`plot/`（字体、通用色板、`CircleLabels`、embedding）。反过来不行。

---

## 现有 `sc_helpers` 文件去向

### 留在层 1（整理后）

| 现在 | 以后 |
| --- | --- |
| `io/_anndata_compat.py` | `io/_anndata.py` |
| `data/_operations.py` | 同名，补 `require_*` |
| `plot/_fonts.py`、`_palettes.py`（函数 + 通用色） | 留；去掉课题色板 |
| `plot/_dotplot.py`、`_grid_dotplot.py`、`_ridgeplot.py`、`_proportions.py` | 留 |
| `plot/_plot1cell*.py` | 留；重复的 geom 以后再抽内部文件 |
| `stats/_stats.py` | 留，不扩张 |

### 迁到 `ext/`

| 现在 | 以后 |
| --- | --- |
| `utils/_drvi_utils.py` | `ext/drvi/_prep.py` + `_plot.py` |
| `plot/_milo.py` | `ext/milo/` |
| `utils/_scenicplus_utils.py`、`plot/_scenicplus_viewer.py`、`_genome_viewer.py` | `ext/scenicplus/` |
| `plot/_contribution_scores.py`、`_locus_panel.py`、`_gene_ccre.py` | `ext/chrombpnet/` |
| `utils/_liana_utils.py`、`data.convert_liana_to_soap` | `ext/liana/` |
| `data.create_mudata_from_ArchProj` | `ext/archr/` |
| `plot/_modality_correspondence.py` | join → `data/_pairing.py`；热图 → `plot/` 或 `ext/multiome/`（若只 multiome 用） |

### 课题仓库留下（层 3）

- `DATA_DIR` / `FIG_DIR`
- `DISEASE_PALETTE`、`CLUSTER_COLORS`、`CLUSTERS_DEFAULT` 等论文色板
- `bulk_helpers` 整包（registry、Fig3 图、队列回归）
- ChromBPNet 的 shell / reporting

色板和细胞类型是跨组学的单一事实来源：同一套 `cell_type_highres` 要同时出现在 RNA UMAP、ATAC UMAP、以后的组织图上。绘图函数接收 `palette: dict[str, str]`，课题色板由分析仓库传入。

### 删除或冻结

- `io/_dask.py`、`io/_zarr.py`（未接线）
- 根目录 `scanpy_helpers.py`
- 根 `__init__.py` 的大 re-export 清单
- `_single.embedding` 与 `_scatterplot.embedding` 两套并存（合并后只留一套对外 API）
- `plot/_spatial.py` 正名：它是 embedding 等高线，不是组织空间；并入 embedding 或改名，避免以后和空间混淆

### 明确暂不建

`pp/`、`tl/`、`spatial/`、`core/`、`logging/`、`datasets/`、`data/_intervals.py`、`data/_mudata.py`、`data/_obs.py`、`plot/_tracks.py`

interval / 多轨布局 / MuData 对齐：两个 `ext` 或两类图开始复制时再抽。不要为「看起来完整」预建空目录。

**判断要不要新基础文件：** 同一段逻辑是否已经被两个以上 `ext` 或两类图复制。复制一次可以忍；复制两次再抽。

---

## 层 1 需要补的小文件（不是新顶层包）

现有 `io` / `data` / `plot` / `stats` 已经够用。缺的是这些目录里还没独立出来、但已经在多处复制的文件。

### 现在就做

1. **`data/_check.py`**  
   `require_obs` / `require_obsm` / `require_var` / `require_layer`  
   几乎每个绘图函数都在重复「列在不在 obs / basis 在不在 obsm」。

2. **`plot/_style.py`**  
   `setup_style()`、`savefig(fig, path)`  
   Arial、`pdf.fonttype = 42`、pdf + png。不要做成可变的全局 `settings` 单例。`_fonts.py` 只负责加载 TTF。

3. **`data/_pairing.py`**  
   跨组学连接是数据，不是图。从 `_modality_correspondence` 挪出 join；热图留在 `plot/`。以后 RNA↔空间映射还是同一套。

4. **色板应用留在 `_palettes.py`**  
   不要另开 `colors/`。`create_palette_from_types`、`reorder_and_set_palettes` 已经是基础。

这四块做完，层 1 闭环：**检查对象 → 配对/过滤 → 上色 → 定字体出图 → 安全写出**。

### 等第二次用到再加

| 文件 | 何时才需要 |
| --- | --- |
| `data/_intervals.py` | 基因组窗口、peak↔gene overlap 被两个 `ext` 共用时。用 `pyranges` / `bioframe`，不要自己写 interval tree |
| `data/_mudata.py` | 需要在 RNA/ATAC modality 间对齐 obs、拷贝列时。ArchR 导入本身属于 `ext/archr/` |
| `data/_obs.py` | 多样本 facet、`library_id` 解析在 RNA 出图和空间出图都出现之后 |
| `plot/_tracks.py` | GenomeViewer 和组织切片 viewer 都写了、开始复制「多轨对齐」时 |
| `io/_tables.py` | 几乎不需要，TSV 用 pandas 即可 |

`library_id` / sample 比空的 `spatial/` 子包更值得先做，但等真正重复时再抽。

### 不要建的「基础模块」

- `pp/` / `tl/`：scanpy / muon / snapatac2 的事
- `logging/`、`config/`、`settings.py`：全局可变状态
- `datasets/`：`simulate_*` 放 `tests/`
- 把 `stats/` 做大：FDR、检验用 scipy / statsmodels；只留改过口径的（如 MAD outlier）
- 为 Visium 再 fork 一份 2000 行 embedding：优先 `scanpy.pl.spatial` / `squidpy.pl.spatial_scatter`

---

## 依赖

```toml
[project]
name = "cellfish"
requires-python = ">=3.11"
dependencies = [
    "anndata",
    "matplotlib",
    "mudata",
    "numpy",
    "pandas",
    "scanpy",
    "scipy",
    "seaborn",
]

[project.optional-dependencies]
plot = ["marsilea", "rich"]
drvi = []              # 按实际包名补
chrombpnet = ["logomaker", "h5py", "pyranges"]
scenicplus = []        # 函数内 import
# spatial = ["squidpy"]  # 第一次真做组织图再加
```

`pip install cellfish` 必须能在只有 RNA 的环境里跑。不要让笼统的 `.[plot]` 把 logomaker 绑进来（它只服务 ChromBPNet）。

---

## 分阶段

### 阶段 0 — 约定（不搬文件）

- 定包名（默认 `cellfish`，import `cellfish as cf`）
- 给每个现有文件打标签：层 1 / `ext/<tool>` / 层 3 / 删除
- README 写清：做什么、不做什么、如何加新算法（复制 `ext/<tool>/` 模板）

包名备注：`selfish` 不可用（PyPI 已占用；GitHub 上还有同领域 Hi-C 工具 [ay-lab/selfish](https://github.com/ay-lab/selfish)）。`cellfish` 的 PyPI 名空着；需注意已发表的 Julia 工具 [CellFishing.jl](https://github.com/bicycle1885/CellFishing.jl)（scRNA 相似细胞检索），检索时可能混。

### 阶段 1 — 瘦 API + 补层 1 缺口（优先）

对后续可扩展性帮助最大、搬家最少：

1. 根 `__init__.py` 只暴露 `io`、`data`、`pl`、`stats`、`ext`
2. 新增 `data/_check.py`
3. 新增 `plot/_style.py`
4. 从 `_modality_correspondence` 抽出 `data/_pairing.py`
5. `pyproject.toml` extras 按算法拆

本仓库 notebook 暂不改 import；可加一层薄兼容（旧 `sc_helpers` 转调新入口）。

### 阶段 2 — 去重层 1 绘图

1. 合并 `_single` / `_scatterplot` 的 embedding 对外 API，统一 `basis=`
2. 正名 `_spatial.py`（等高线 ≠ 组织图）
3. `_palettes.py` 移出课题色板，改由本仓库传入 `palette=`
4. 删 dask/zarr 桩和 `scanpy_helpers.py`
5. 新图函数一律走 `require_*` + `setup_style` / `savefig`，禁止再复制 font loader

### 阶段 3 — 按算法进 `ext/`（可分 PR，每个工具一次）

建议顺序（依赖从少到多）：

1. `ext/milo/`
2. `ext/liana/`
3. `ext/archr/`
4. `ext/drvi/`（先拆 `_prep` / `_plot`，这是最大一块）
5. `ext/scenicplus/`
6. `ext/chrombpnet/`（contrib / locus / cCRE 同文件夹、分文件）

每迁一个：公开 API 从 `cf.ext.<tool>` 出；smoke test 跟着走；层 1 不得反向依赖。

### 阶段 4 — 本仓库接上

- `MASLDHCC_reproducibility`：`pip install -e ../cellfish`，或暂时仍放 `src/` 但包名换成 cellfish
- 薄封装保留旧 `import sc_helpers as sch`（可选，避免一次改完所有 notebook）
- `bulk_helpers` 删掉 `sys.path` hack，改为 `import cellfish as cf` + `cf.pl.setup_style()`
- 课题色板、路径放 `project.py`（或现有 constants），不进 cellfish

### 阶段 5 — 独立仓库（第二个课题要用时）

- 新 git repo，保留 src layout + extras + 少量 smoke test
- 夹具：`rna_tiny` / `atac_tiny`（peak var）/ 以后 `spatial_tiny`（`obsm["spatial"]`）
- 确认层 1 的 embedding / dotplot 在三类对象上都能跑（空间对象只换 `basis="spatial"`）

### 阶段 6 — 空间（有真实分析再开）

- 不建 `ext/spatial/` 大筐
- 通用组织散点：`cf.pl.embedding(..., basis="spatial")` 或薄包一层 scanpy / squidpy，禁止再 fork 2000 行
- 算法（Cell2location、Tangram…）各建 `ext/<tool>/`
- extra：`spatial = ["squidpy"]`
- 若 GenomeViewer 和组织 viewer 开始复制多轨布局 → 再抽 `plot/_tracks.py`

---

## 加新算法的固定模板

1. 新建 `ext/<tool>/__init__.py` + 一个实现文件
2. 计算和出图放一起；颜色 / 字体 / grid 调 `cf.pl`
3. 可选依赖放 pyproject extras + 函数内 import
4. 加一条 smoke test（可模拟小对象）
5. 不要改层 1，不要新建 `plot/_<tool>.py`

## 验收标准

- `import cellfish as cf` 不要求 logomaker / h5py / marsilea
- 层 1 无 `from cellfish.ext ...`
- 课题色板、路径不在包内
- 每个 `ext/<tool>` 可独立缺依赖失败，并提示 extra 名
- 本仓库至少 1 个 RNA notebook、1 个 ATAC 出图 notebook、`bulk_helpers` 能不靠 `sys.path` 工作
- 没有第二套 embedding 实现、没有未使用的 io 桩

## 建议的第一步

做**阶段 1**：瘦 `__init__.py`、补 `data/_check.py` 和 `plot/_style.py`、pairing 挪位、extras 按算法拆。文件基本不搬家，但边界已经锁死，后面迁 `ext/` 不会来回改。
