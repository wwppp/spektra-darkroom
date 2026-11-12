# 工作视口色域切换方案（参照 Photoshop / Lightroom / Capture One）

状态：**方案稿，未落地**。日期：2026-11-12。适用版本：v0.6.1。

---

## 0. 结论先说

**能做，而且不该做成"工作色彩空间下拉框"。**

专业修图软件把颜色拆成**四层**：文档（工作）空间、输出/校样空间、显示空间、渲染意图。
我们现在的视口把这四层压成了一层"固定 sRGB"，所以才有 P3 屏偏色、导出色域与预览脱节两个问题。

正确的落地顺序是：

| 期 | 做什么 | 解决什么 | 是否动渲染域 |
|---|---|---|---|
| **P0** | 显示端色彩管理（display transform） | P3 / Adobe RGB 屏不再偏艳 | 否，只动 shader 末段 |
| **P1** | 校样（soft proof）+ 色域警告 | 「导出成 sRGB / Adobe RGB 会变成什么样」看得见 | 否，串在显示端之前 |
| **P2** | 输出/工作色域可选（P3 直出、10-bit、抖动） | 真吃到广色域、消灭色带 | **是**，动链尾 + 缓存 key |

"让用户像 PS 一样切工作空间"（sRGB ↔ Adobe RGB ↔ ProPhoto）**不建议做**，理由见 §2。

---

## 1. 参照系：专业软件到底怎么分层

| 概念 | Photoshop | Lightroom | Capture One | 我们对应的东西 |
|---|---|---|---|---|
| 文档空间 | 工作空间（建议 ProPhoto/16-bit） | 内部固定 ProPhoto 线性 | 内部固定（相机 ICC） | **内部 = 场景线性 ProPhoto → Rec.2020**（物理链，见 §2） |
| 输出/校样空间 | 校样设置（Ctrl+Y） | Soft Proofing（S 键） | Proof Profile | **P1 要做的东西** |
| 显示空间 | 由 OS + 显示器 ICC 补偿 | 同 | 同 | **P0 要做的东西**（现在缺） |
| 渲染意图 | 可感知 / 相对比色 + 黑场补偿 | 同 | 同 | P1 选项目 |
| 色域警告 | Ctrl+Shift+Y | 越界高亮 | 有 | P1 的副产品（clip 掩膜本来就有） |

关键点：**PS 让用户切的是"文档空间"，而文档空间的自由度来自"它要接收任意来源的图像"。我们的图像不是任意来源——它是一条物理显影链的产物，色域由染料光谱决定。** 所以照搬 PS 的"工作空间"概念是错位的，照搬它的"校样 + 显示补偿"才是对的。

---

## 2. 为什么"工作空间"不让我们切

现在链的实际形态（已逐行核对）：

```
SCENE-LINEAR ProPhoto (raw_preview, float)
  ├─ ×exp2(base_ev)                     canvas_viewport.py:982
  └─ M_PP_TO_2020                       gl_ffchain.py:42
REC.2020 → 色度归一化 → 曝光表 → logE → 胶片曲线 → DIR → 净密度
  → filmOut 表 → 相纸曲线 → paperOut 表 → 线性 P3   gl_ffchain.py:324
线性 P3 → M_P3_TO_SRGB → clamp(0,1) → sRGB 编码    gl_ffchain.py:328-331
```

CPU 导出侧同构：`core/ffchain_point.py:288-290` + `srgb_encode()`（内含 `np.clip(v,0,1)`）。

三个事实决定了"工作空间"没有可切性：

1. **相纸的输出天然是 P3**（`paperOut` 表的通道就是线性 P3）。sRGB 是最后一步矩阵压出来的，不是工作空间。
2. **往前切没有物理含义**：ProPhoto/Rec.2020 这两级是光谱计算的中间表示，换个"工作空间"等于换算法，不是换设置。
3. **往后切是自欺欺人**：源已经在 sRGB 被 clip，导出成 ProPhoto 只是换容器（详见 §5 的现状诊断）。

所以真正可切的只有两样：**显示用哪个色域**（P0）、**校样看哪个色域**（P1）。

---

## 3. 现状诊断（为什么必须做，不做会怎样）

| 现象 | 根因 | 位置 |
|---|---|---|
| P3 / Adobe RGB 屏上预览偏艳 | 输出未标记的 sRGB 数值，无人做显示补偿 | 全仓无 `QSurfaceFormat` / ICC 读取 |
| 导出色域与预览"看起来一样" | 渲染源恒为 sRGB，转换源也是 sRGB（`MATRICES` 是 sRGB→目标） | `core/colorspace.py:126-140` |
| 广色域能力被丢弃 | `lin_p3` 已有，被 `M_P3_TO_SRGB` + clip 压掉 | `gl_ffchain.py:328`、`ffchain_point.py:289` |
| 主视图与占位图在广色域屏上不一致 | 旁路贴图是 sRGB 直通，未标记 | `canvas_viewport.py:1020-1022` |

---

## 4. P0：显示端色彩管理

### 目标
把"线性 P3 画面"正确送到**这台显示器**上。用户不需要选任何东西——读系统绑定给显示器的 ICC，自动做。

### 数据流（改后）

```
线性 P3 ──(P1 校样，可选)──> 线性 P3' ──M_display──> 显示器线性 RGB
                                             └─ 显示器 TRC 编码 ─> 上屏
```

`M_display` = `M(display→XYZ)⁻¹ · M(P3→XYZ)`。若显示器 profile 恰好是 sRGB，`M_display` 为**单位阵**（除量化外逐字节不变），这是最强的回归判据。

### 落点

| # | 改动 | 文件 |
|---|---|---|
| 1 | 新增 `core/display_profile.py`：读显示器默认 ICC + 解析 matrix/TRC 标签 → `(3×3, TRC, whitepoint)`；提供 `identity_if_unknown()` 兜底 | 新文件 |
| 2 | ICC 解析器（只支持 matrix/TRC 型，覆盖 99% 显示器 profile；LUT 型降级为单位阵并记日志）| 同上 |
| 3 | 取默认 profile：`mscms.dll::WcsGetDefaultColorProfile`（per-monitor），失败回落到 `%WINDIR%\System32\spool\drivers\color\sRGB Color Space Profile.icm` | 同上 |
| 4 | shader：把尾段拆成 `vec3 to_display(vec3 lin_p3)`（矩阵 + TRC 编码），替换现有硬编码 `M_P3_TO_SRGB` + sRGB 编码 | `gl_ffchain.py:327-331` |
| 5 | 新增 uniform `u_display_matrix`（mat3）、`u_display_trc`（int，0=sRGB 1=gamma 2.2/2.199 2=纯 2.4），上传点同 `u_ff_p3_to_srgb` | `gl_ffchain.py:583` 附近 |
| 6 | Before / 旁路 sRGB 贴图也走 `to_display`：`u_is_bypass==2` 分支先 `srgb_decode` 再 `to_display` | `canvas_viewport.py:1020-1022`、`:983` |
| 7 | 首选项 →「图像与渲染」加一条：`视口色彩管理`（自动 / 关闭），默认自动；关闭=回到今天的恒 sRGB 行为（给 ACM 用户/排查用） | `preferences_dialog.py` |
| 8 | 显示器切换 / profile 变化时重读（`QWindow.screenChanged` + 每 N 秒 or 焦点回来时比对 profile 路径与 mtime） | `canvas_viewport.py` |

### 不做的事（避免过度设计）
- 不请求 10-bit 帧缓冲：Qt + WGL 上不稳，收益低；色带另用抖动解决（P2）。
- 不做逐屏不同 profile 的多窗口：当前是单窗口。

### 必须处理的冲突：Windows ACM

Win11「自动管理颜色」开启时，系统**会把未标记窗口当 sRGB 转到显示器 profile**。我们若再自己转一次 = 双重转换，颜色反而错。
处理：启动时检测 ACM 状态（`DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO` 的 `advancedColorSupported/Enabled`，或注册表兜底）；ACM 开 → 默认"关闭自管"并把首选项那一项标灰 + 解释文案；ACM 关 → 默认自动自管。**这条是 P0 的唯一硬阻塞点，必须先定。**

---

### 4.1 落地结果（P0 已完成）

| 文件 | 内容 |
|---|---|
| `core/display_profile.py`（新，约 640 行） | ICC 解析（矩阵 + 3 类 TRC）、`identity()` 兜底、`current()`、`acm_state()` / `managed_default()`、与 lcms 对拍用的纯 numpy 参考实现 |
| `ui/gl_ffchain.py` | 尾段 `#5` 改成「矩阵 → 可选纯 gamma → sRGB 编码」；新增 4 个 uniform；`set_display()` / `sync_display()` / `_bind_display()`；`bind(..., display=True)` |
| `ui/canvas_viewport.py` | Before 分支、`u_is_bypass==2` 分支、`develop_negative` 末尾（老 LUT / 无胶片）三条旁路都走 `ff_display_encode`；`render_offscreen(display_transform=False)`；3 秒轮询 + 首选项开关 |
| `ui/preferences_dialog.py` | 删掉死控件「工作色彩空间」，换成「视口色彩管理」+ 一行真实状态（当前 profile / 为何不接管） |
| `ui/main_window.py` | 关掉首选项对话框后立刻重读偏好（`_apply_display_cm_pref`），不必重启 |

**关键实现决定（与本文原计划不同的地方）**

1. **不新增 `u_display_matrix`**：`u_ff_p3_to_srgb` 在 CPU 侧直接被换成合成的「P3 → 显示器」矩阵，另有 `u_ff_srgb_to_display` 只给线性 sRGB 入口（Before / 老 LUT）用。点式链尾段因此只占一个 mat3。
2. **表格型 TRC 降级为「不接管」**：Intel 驱动上 `uniform vec4 curve[64]` + 动态下标会直接段错误（`texture()` 查表版本同样），所以只保留解析式（sRGB 分段 + 纯 gamma）覆盖；`TRC_TABLE` 返回 `None` → `identity` 语义。Rec.2020 profile 就是这种，实测降级。
3. **ACM 探测的两种失败都不会"看起来正常"**：见下面两个坑。

**两个真实踩坑（都不是逻辑错，是结构体尺寸）**

- `DISPLAYCONFIG_PATH_INFO` = **72** 字节（`sourceInfo` 20 + `targetInfo` 48 + `flags` 4）。少写 `targetAvailable`/`statusFlags` 就变 64 → `QueryDisplayConfig` 仍回 0，但按 64 读 72 写的缓冲区**越界 8 字节/条**，堆被踩 → 第一次调用正常、之后随机段错误。
  `sourceInfo` 里 `modeInfoIdx` 的 16+16 是它**自己的联合体**（cloneGroupId / sourceModeInfoIdx），不是 `statusFlags`；把两者合成一个 32 位位域同样少 4 字节。
- `DISPLAYCONFIG_MODE_INFO` = **80** 字节（`infoType` 4 + `id` 4 + `adapterId` 8 + 联合体 64），按 64 分配同样越界。
- 症状极具误导性：尺寸错时 `DisplayConfigGetDeviceInfo` 回 `ERROR_INVALID_PARAMETER(87)` → 读成"系统不支持 ACM" → P0 静默永不生效。现在 `_user32()` 里加了 `_check_struct_sizes()`，导入期就报错。

**验收（`scratch/` 四个脚本，全绿）**

| 脚本 | 判据 |
|---|---|
| `_display_p0_check.py` | identity 恒等（矩阵与 `M_P3_TO_SRGB` max\|d\|=0）；5 套 ICC 与 lcms 对拍中位 0 / 最大 ≤2；TRC 解析分类正确；降级路径 |
| `_display_p0_stages.py` | 导出三态逐字节相同（max\|d\|=0）；identity 时 view==export；P3 屏 view 差 98 级而 export 仍为 0 |
| `_display_p0_ui.py` | 死控件已删、开关往返、`ffchain.display` 随开关变 |
| `_display_p0_soak.py` | 3 秒轮询 8 秒不重建 HQ、状态行稳定、开关三次来回 |

对拍脚本排除「色域外样本」是**策略差异不是算错**：lcms 做色域映射，我们 clamp。

**P0 已知残余（写在这里，免得下次当 bug 查）**

- 缩略图 / 直方图仍走 sRGB 域（CPU 侧 `prophoto_to_srgb_filmic`）。广色域屏上缩略图会比主视图淡——和「直方图按文档空间算」是同一类取舍，不是漏接。
- 表格型 TRC（Rec.2020 这类）在 Intel 驱动上只能降级为不接管；换驱动后可以再试一次查表版。
- 单窗口单屏。多屏同时不同 profile 不做。
- 每次 3 秒轮询只比对「profile 路径 + mtime」，所以换显示器必须系统真的换了绑定文件才触发重读；换屏但同 profile 时不重建 HQ（本来就是对的）。

---

## 5. P1：校样（soft proof）

### 目标
对齐 PS 的 Ctrl+Y / LR 的 S 键：**"如果输出成 X 会变成什么样"**，并让导出与它同源。

### 语义
- 校样目标 = `resources/icc/` 里那五套之一（sRGB / Display P3 / Adobe RGB / Rec.2020 / ProPhoto），外加将来的 CMYK/纸白 profile。
- 渲染意图：默认**相对比色 + 黑场补偿**（PS 默认）；可选**可感知**（用 Oklab 色度衰减实现，不用自己写 CIECAM）。
- 纸白模拟（"模拟纸墨"）：把目标白点乘进画面——这是印刷校样的关键，屏幕看白纸会偏灰。
- 色域警告：复用 clip 掩膜（`lin_srgb` 越界或目标色域越界），叠加品红/灰斑。

### 数据流

```
线性 P3 ─> [P1] M_proof(目标线性) ─> gamut_map(越界处理) ─> M_proof⁻¹ ─> 线性 P3' ─> P0 显示端
```

中间回 P3 是为了让 P0 保持不变；也可以直接串到显示器矩阵，省一次矩阵（推荐，但要把两者合成一个 mat3 上传，避免每次改校样都重编译 shader）。

### 落点
1. `core/colorspace.py`：把 `MATRICES`（现在只有 sRGB→目标）补上**目标→sRGB 的逆**，以及 P3→目标、目标→P3；统一成"任意两空间互转"的小表。
2. 新增 gamut mapping：Oklab OKLCh chroma 二分（cut-to-gamut），`core/colorspace.py` 内，纯 numpy 可测；GPU 端用 20 次迭代的近似版（或烘成 1D LUT 沿 chroma 轴）。
3. shader：`u_proof_on`、`u_proof_matrix`、`u_proof_intent`、`u_proof_paper_white`、`u_proof_warn`。
4. UI：`视图` 菜单 → 「校样设置…」对话框（目标 / 意图 / 模拟纸墨 / 色域警告），`Ctrl+Y` 开关、`Ctrl+Shift+Y` 警告开关——**和 PS 一致**，用户不用重学。
5. 导出联动：导出对话框的 `colorspace` 与校样目标**双向同步**（选了校样 sRGB，导出默认也 sRGB；导出换了色域，校样跟着换）。这是"所见即所得"的闭环，也是现在最缺的一环。
6. 缓存：校样只影响显示，**不进 LUT key**（LUT 仍烘 sRGB），所以切换校样 = 重绘一帧，零重建。

### 5.1 落地结果（P1 已完成，范围已收窄）

**范围拍板（2026-11-12）**：只做 5 套 RGB 目标 + 相对比色（= clip）+ 品红叠加警告 +
导出双向联动。**不做 CMYK / 不做纸白模拟 / 不做可感知意图**，理由：

- 印刷校样真正要的三样（小色域、GCR/黑版、油墨量）都编码在 CMYK profile 的 A2B/B2A LUT 里，**不是矩阵**。硬做只能得到"看起来像校样"的东西。
- GPU 端要新增多维 LUT 采样，而 P0 已经在 Intel 驱动上为表格型 TRC 让过路（`uniform` 数组动态下标直接段错误）。用最高风险的路去覆盖最小众的需求。
- 可感知意图（Oklab 色度衰减）与导出的 clip 不一致，会重新制造"预览一个样、文件另一个样"——正是 P1 要消灭的问题。想做得和导出同源，导出那侧也必须上同一套映射，那是独立一轮工作。

| 文件 | 内容 |
|---|---|
| `core/proof.py`（新） | `ProofSettings`、`matrix_p3_to_target` / `matrix_target_to_p3`、`compose_display`（把回程矩阵折进显示器矩阵）、`out_of_gamut` / `gamut_coverage` 参考实现 |
| `ui/gl_ffchain.py` | 尾段在 `lin_p3` 之后插入 `u_proof_matrix` 一次矩阵 + 一次 clamp；`u_proof_on` / `u_proof_warn` / `u_proof_eps`；`set_proof()` / `_proof_state()` |
| `ui/canvas_viewport.py` | `proof_settings()` / `set_proof()` / `proof_status()`；`proofChanged` 信号；切换时延后作废 HQ 纹理 |
| `ui/proof_dialog.py`（新） | 「校样设置」：开关 + 目标 + 色域警告 |
| `ui/main_window.py` | 视图菜单三项；`Ctrl+Shift+Y` 校样、`Ctrl+Alt+Y` 警告；`_sync_export_colorspace` / `_sync_proof_from_export` 双向联动 |

**关键实现决定**

1. **只多一个 mat3**：`compose_display(base, settings) = base @ M_target→P3`，shader 里再乘一次
   `u_proof_matrix` 就完成 P3→目标→clip→P3 的往返。回程矩阵不上传，省一个 mat3（shader 越大
   Intel 驱动越危险）。
2. **越界判定的容差 `GAMUT_EPS = 1/512`**：矩阵往返本身有 ~2e-4 的舍入（尾段常量
   `M_P3_TO_SRGB` 当年被手工抹过小项），恰好落在 0/1 上的像素会被判越界，画面出现一片没有
   意义的品红麻点。1/512 相当于半个 8-bit 量化级。
3. **快捷键让位**：`Ctrl+Y` 保留给「重做」（Windows 肌肉记忆，用户明确要求），校样改
   `Ctrl+Shift+Y`、警告 `Ctrl+Alt+Y`。PS 的 `Ctrl+Y` 习惯在这里让位给已有的撤销栈。
4. **校样只作用于胶片链输出**：Before 面板 / 占位图 / 老 LUT 出口不校样（它们是参考物，没有
   色域承诺）。这是已知取舍，不是漏接。

**验收（`scratch/`，全绿）**

| 脚本 | 判据 |
|---|---|
| `_proof_p1_math.py` | 校样总变换 ≡ 链尾矩阵（max\|d\|=2.2e-16，即"与导出同源"）；界内 roundtrip 恒等；越界掩膜与可见变化一致（漏报 0）；四套目标与 colour-science 差 ~2e-4（尾段常量的手改量） |
| `_proof_p1_stages.py` | 导出在 5 种校样状态下逐字节相同；校样=sRGB 时 view==export（max\|d\|=0，闭环）；校样=Adobe 有像素被夹；校样=P3 几乎不变；GPU 警告像素与 CPU 掩膜逐像素吻合（25/25，交集 23，差在 8-bit 量化）；警告像素 100% 品红方向 |
| `_proof_p1_ui.py` | 快捷键不冲突（`Ctrl+Y` 仍是重做）；菜单勾选与 canvas 同步；对话框往返；校样→导出、导出→校样两个方向；校样关闭时不改写导出设置 |

**P1 已知残余**

- 校样不开在 Before 面板上（见上）。
- 可感知意图 / CMYK / 纸白未做（理由见上）。
- 导出仍是 clip，和校样一致；若将来导出改走 Oklab 映射，校样必须同步改，否则闭环断掉。

---

## 6. P2：真吃到广色域（已完成 2026-11-12）

原计划（下面 1–7 条）是照"视口和导出都要改"写的。落地时发现**视口那半边 P0 已经
顺带做完了**：`u_ff_p3_to_srgb` 在显示管理接管后就是「P3 → 显示器」的合成矩阵，校样
关闭时 shader 只是 `max(lin_p3, 0)` 之后直接送屏，clamp 夹的是**显示器**边界而不是
sRGB。所以 P2 只剩导出侧——`rendered_float` 那份"假定 sRGB"的假设。

### 6.1 先量收益（`scratch/_p2_gamut_audit.py`）

均匀采样（ProPhoto 线性色度三角形内均匀，20 万点）会给出很吓人的数字：45–58% 越界。
那批点全在三角形边上，是极端色。真实照片（`测试/金200/金200.jpg`）跑同一条链才说明问题：

| 卷 | 越界像素 | >1/255 可见 | 中位越界 |
|---|---|---|---|
| harman_phoenix_200 | 12.85% | **7.91%** | 0.0063 |
| kodak_aerochrome_1443 | 36.69% | **10.67%** | 0.0006 |
| fujifilm_superia_200 | 3.60% | 0.84% | 0.0015 |
| fujifilm_pro_160ns | 0.41% | 0.13% | 0.0020 |
| fujifilm_velvia_50 | 1.29% | 0.01% | 0.0008 |
| cinestill_800t | 0.43% | 0.01% | 0.0004 |

平均 3.26% 可见越界，且**高度集中在少数卷**。结论没变：P2 值得做，但画面差异只出现在
phoenix / aerochrome 这类本身很饱和的卷上，其余卷的收益是"文件标签与像素一致"。

### 6.2 落地

| 文件 | 内容 |
|---|---|
| `core/colorspace.py` | `convert(..., source_colorspace="sRGB")`；`M_P3_TO_SRGB` 直接从 `ffchain_point` 导入（不是抄一份），保证导出解的就是链尾乘的 |
| `core/app_core.py` | `save_encoded_image` 按 `params["source_colorspace"]` 决定转换源；`sRGB→sRGB` 的早退只在两侧都真是 sRGB 时才走 |
| `ui/gl_ffchain.py` | `bind(..., output_p3=, dither=)`；`_bind_display` 里 `output_p3` 时矩阵换单位阵；尾段加 TPDF 抖动（`u_out_dither`，默认 0） |
| `ui/canvas_viewport.py` | `render_offscreen(..., output_p3=)`；新增 `last_offscreen_source_space()` |
| `ui/export_queue_manager.py` | 目标=P3 时请求 `output_p3=True`，保存前用**画布回报的实际空间**填 `source_colorspace` |

**四个必须记住的决定**

1. **`output_p3` 只给 Display P3，不给其它广色域。** Adobe / Rec.2020 / ProPhoto 的
   传递函数各不相同，链尾只有 sRGB 那套编码，要支持就得再给 shader 加编码分支——
   Intel 驱动已经贴着上限（P0 的教训）。Display P3 与 sRGB **共用分段 TRC**，所以
   "输出 P3"就等于"矩阵换单位阵"，零成本。
2. **标记必须来自画布的回报，不能来自"我请求了 P3"。** 点式链没接管（原生卷 / 老 LUT
   出口）时请求会被静默退回 sRGB，按请求标记就是 sRGB 像素 + P3 标签，整张图偏色。
   `last_offscreen_source_space()` 就是为这件事存在的。
3. **源=目标必须直接返回。** P3 源导出 P3 是主路径；漏掉这步会让
   `eye @ M_P3_TO_SRGB` 把 P3 悄悄转成 sRGB 再标成 P3。
4. **抖动只对 8-bit，且放在色域警告之后。** TPDF = 两个均匀量之差，幅度 1 LSB。

### 6.3 验收（`scratch/`，全绿）

| 脚本 | 判据 |
|---|---|
| `_p2_export_check.py`（11 项） | 源=sRGB 时 5 个目标空间的结果与旧实现一致（max\|d\|≤1.7e-4）；未知源按 sRGB；P3→sRGB 误差 0.03 LSB；P3→P3 恒等；P3→Adobe 与手算一致 |
| `_p2_gpu_check.py`（7 项，分进程） | 标记正确（含原生卷退回 sRGB）；**闭环：P3(16bit) 转回 sRGB == 同帧 sRGB 导出**，max\|d\|=1.06 LSB、mean 0.21 LSB；P3 帧里确有 sRGB 外颜色；抖动不偏色（0.07 LSB）且幅度≈1 LSB（0.36，理论 0.33） |
| 逐字节回归 | 同一帧改前/改后：8-bit **max_abs=0**、16-bit **max_abs=0** |
| `_proof_p1_stages.py` | P1 全部判据仍全绿（校样闭环 max\|d\|=0） |
| `_display_p0_stages.py` | P0 三态导出逐字节相同 |

### 6.4 已知残余

- **CPU 回退路径仍是 sRGB。** GPU 失败回落 CPU 时 `params` 里没有 `source_colorspace`，
  按 sRGB 处理——文件会比 P3 窄，但不会撒谎。
- 抖动只在 8-bit 开；16-bit / float 输出的量化台阶远低于可见阈值，不需要。
- 广色域目标仍从 sRGB 转（见决定 1），所以只有 Display P3 是"真原生"。
- 老 LUT 出口（原生卷，如 gold200）不走点式链，导出 P3 时仍是 sRGB 内容。

---

## 6bis. 原 P2 计划（留档，已被上面替代）

只有用户明确要"P3 原生输出"时才做。代价最大，收益是导出文件真含广色域颜色。

1. `ffchain_point.print_net` 增加 `output_space="sRGB"|"P3"`，P3 时跳过 `M_P3_TO_SRGB` 且**不 clip**（改为 clip 到 P3 边界）。
2. GPU 同构（`gl_ffchain.py` 尾段）。
3. **导出源标记必须同步**：`core/app_core.py:941-960` 现在假定 `rendered_float` 是 sRGB；改成按 `render_output_space` 决定转换源（P3→目标），否则双重转换，颜色全错。
4. LUT 缓存 key 加 `output_space`（`_lut_cache_key`，`app_core.py:2712` 附近）；`ffp1__` 盘缓存同理。
5. 显示端：内容线性 P3 → P0 已能正确送屏，无需额外改动。
6. 精度：8-bit 广色域渐变会出带。加 TPDF 抖动（shader 里 1 LSB 噪声，成本≈0），或评估 10-bit。
7. 回归：gold200 / 反转片 / 颗粒 σ 全部要重跑——因为输出域变了，`_hq_fit_check` 那类"逐字节相同"的判据需要重新定基线。

**实际落地与原计划的差异**：第 1、4 条不需要（导出走点式链的 GPU 路径，不查 LUT，
`print_net` 只服务缩略图/CPU 侧的 LUT 烘焙）；第 5 条确认成立且**视口早就可以了**；
第 7 条不需要重跑——默认 sRGB 导出逐字节不变，只有用户显式选 P3 才走新路。

---

## 7. 验收判据（可执行，不接受"看着对"）

| 判据 | 方法 | 阈值 |
|---|---|---|
| P0 恒等 | 显示器 profile = sRGB 时，改动前后同一帧逐字节对比 | **完全一致**（矩阵为单位阵，编码路径不变） |
| P0 数值 | 5 套内置 ICC + 系统 sRGB，转换结果 vs lcms/colour-science 参考 | 中位 ≤1 LSB，最大 ≤2 LSB |
| P0 无缓存污染 | 换显示器 profile 后不重建 LUT（`ffp1__` 不新增/不改 mtime） | 0 次重建 |
| P1 闭环 | 校样=sRGB 时，视口截图 vs **导出 sRGB 文件**，在 sRGB 屏上逐像素 | ≤1 LSB（8-bit 往返）。**实测 max\|d\|=0** |
| P1 警告 | 色域警告掩膜 vs 目标色域 clip 掩膜 | 100% 重合。**实测 GPU 25 px / CPU 25 px，交集 23（差在 debug 通路的 8-bit 量化）** |
| 回归 | grain σ / 反转片对拍 / HQ 预览对拍 / 导出逐字节 | 与 v0.6.1 基线一致（P0/P1 不改渲染域，必须全绿） |
| 性能 | 1600×1000 单帧 paint 时间 | 增量 ≤3 ms（P0 一次 mat3 + TRC；P1 若用迭代 gamut map 需实测，超了就限流到"校样打开时降级为 clip"） |

**反例判据（必须写进脚本，避免自欺）**：不能只在 sRGB 屏上验收 P0——那样恒等判据自动成立、看不出任何东西。必须用一个非 sRGB 的 profile 做数值对拍。

---

## 8. 风险清单

1. **ACM 双重转换**（§4）。唯一硬阻塞，先定策略再写代码。
2. **占位图/Before 面板空间标记**：忘记标会导致广色域屏上"占位图比主视图艳"，用户会以为是渲染 bug。
3. **直方图仍是 sRGB 域**：PS 也是按文档空间算直方图，可接受，但要在文档里写清楚，否则会被当成不一致。
4. **HQ 预览（`render_offscreen`）会把显示 transform 烘进离屏图**：profile 变化必须 invalidate，否则换屏后那张缓存图还是旧色彩。
5. **LUT 型显示器 profile 不支持**：只能降级单位阵 + 日志；要在首选项里显示"当前 profile 类型"。
6. **校样与导出脱钩**：P1 若不同步导出对话框，会出现"校样是 Adobe RGB、导出是 sRGB"，比现在更误导。
7. **CPU 侧一致性**：缩略图（`thumbnail_controller.py:181/463`）走 `prophoto_to_srgb_filmic`，P0 后它们也该走显示端，否则缩略图与主视图在 P3 屏上不一致。

---

## 9. 建议的落地顺序

1. 先定 ACM 策略（读状态 + 决定默认行为）——半天，决定后面所有设计。**已完成**（ACM 开 → 不自管）。
2. P0：profile 读取 + 解析 + shader 尾段重构 + 恒等/数值两套判据。**已完成**（见 §4.1）。
3. P0 收尾：占位图、Before、缩略图三处空间标记。**占位图 / Before 已完成**；缩略图按 §4.1 残余保留在 sRGB 域。
4. P1：校样设置 UI + gamut map + 导出联动 + 闭环判据。**已完成（范围收窄，见 §5.1）**。
5. P2：导出侧 P3 原生 + 抖动。**已完成**（视口侧 P0 顺带完成，见 §6）。

---

## 10. 一句话给用户看

改完之后你可以：
- 用 P3/广色域屏时颜色不再偏艳，和 Photoshop 里看到的对得上；
- 按 `Ctrl+Shift+Y` 看"这张图导出成 sRGB / Adobe RGB 会变成什么样"，越界的颜色会被标出来
  （`Ctrl+Alt+Y` 单独开关警告）；
- 导出的色域和校样自动一致，不再出现"预览一个样、文件另一个样"；
- 导出选 **Display P3** 时，文件里那些超出 sRGB 的颜色是真的保住了（phoenix / aerochrome
  这类饱和卷最明显），不是"sRGB 内容套了个 P3 标签"。
