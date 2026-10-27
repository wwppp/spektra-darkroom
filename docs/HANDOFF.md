# 交接文档（HANDOFF）

> 用途：把当前工作从旧会话交接给新会话。**新会话只需读这一个文件 + `docs/grain-rework-plan.md`**，不需要旧对话。
> 最后更新：2026-10-27，v0.4.54 / Build 216

---

## 0. 三十秒上手

```bash
cd D:\spektra-darkroom
python tools/validate_fotufilm_port.py --quiet     # 759 项档案校验，应全 PASS
python tools/grain_parity.py                       # 33 项颗粒一致性，应全 PASS
python tools/grain_stats.py                        # 7 项统计目标，应全 PASS（PSD 一行为 INFO）
python tools/export_colorspace_check.py            # 41 项色彩空间
python tools/test_export_features.py               # 12 项导出格式
python scratch/dir_probe10.py                      # 全功能冒烟，看 "python exceptions: 0"
```

（PowerShell 下 `$env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONPATH="D:\spektra-darkroom"` 先设好；末行 `[exit code: 1]` 是已知的 harness 收尾现象，判定看 `RESULT: PASS`。）

---

## 1. 工作区状态

- 版本 **0.4.54 / Build 216**（`core/version.py`）
- **尚未提交**：76 个改动/新增文件。建议转移前先 `git add -A && git commit`，否则一旦误操作很难回退。
- 新增未跟踪：`core/colorspace.py`、`core/grain_model.py`、`ui/tooltip.py`、`docs/`、`tools/` 下多个脚本

---

## 2. 已完成（全部有脚本验收）

| 阶段 | 内容 | 版本 |
|---|---|---|
| **P1b** | 导出色彩空间单一解析器 + numba 内核 + ICC 失败语义；修 ProPhoto D65→D50 缺色适应（会偏色）；64 MP 非 sRGB 从 30–38 s 降到 0.2–0.9 s | 0.4.46 |
| **P0** | 用 spektrafilm 粒子模型做物理参考（8→10 档密度 × 3 档尺寸），拟合颗粒度-密度钟形曲线，产出 `tools/grain_params.json` | 0.4.48 |
| **P1** | `core/grain_model.py` 成为唯一真源，消除预览/导出 2.5× 振幅差；σ 下限保护 | 0.4.47 |
| **P2** | 密度响应换标定钟形；通道相关按目标反解 rho；振幅改物理口径（∝ 长边） | 0.4.49 |
| **性能** | 颗粒合成换 numba RNG + 解析高斯方差（4K 1975→1165 ms，24 MP 8912→6115 ms） | 0.4.50 |
| **数据层** | 转换档案补 4 字段；`density_law` 修 27/33；原生 20 支 law 修正；`cinestill_400d` 曲线重算；Instax 分类修正 | 0.4.52–0.4.54 |

**已取消**：P3′（异步 CPU 颗粒场）、P4′（CPU 真 MC）。理由见 `docs/grain-rework-plan.md`——实测 4K 颗粒场 1165 ms（预算 300 ms），粒子模型 24 MP 49 s（预算 10 s）；计划里的性能数字是估算、从未测量。

---

## 3. 待办一：每胶卷密度律接线（**这是唯一还欠的渲染改动**）

### 为什么必须做
`crystal_grain` 里的 `density_law` / `density_profile` / `luma_correlation` / `granularity_read_density` / `fog_density` / `reversal_profile` **已写进档案、已透传到 `get_film_grain_preset()`，但渲染器还没消费**。现在所有胶卷共用 P0 标定的**一条全局**钟形曲线和全局相关目标 0.79。

### 四种密度律（源码依据：`fotufilm-engine/Sources/FotufilmCore/FilmStock.swift`）

```
law 0 dyeCloud      : σ²(D) ∝ (1−e^(−D/toe))·(1 + A·e^(−D/decay) + H·e^(−(D−hD)²/(2·hW²)))
                      一行 6 系数 [A, toe, decay, H, hD, hW]，RGB 各一行
law 1 silver        : σ²(D) ∝ D·10^(0.21004·D + 0.06114·D²)
law 2 dyeCloudSelwyn: σ_D ∝ √D
law 3 dyeCloudReversal: σ²(D) ∝ D^(2p)/(1 + (D/Ds)^(2p))，p,Ds 来自 reversal_profile（默认 [1.1, 3]）
```

归一化（`FilmStock.swift:611` 的 `grainDensityModulation`）：

```
anchor = granularityAnchorDensity(layer)
       = granularity_read_density=="gross" ? (1 − dMin) : 1
         再 clamp 到 [0.05·range, 0.9·range]，range = dMax − dMin
here   = max(netDensity, 0) + fog_density
mod(layer,D) = sqrt( V(here) / V(anchor + fog_density) )
```

最终 `dens_mod = DENS_GAIN × mod(D)`，其中 `D = −log10(clamp(lin, 1e-4, 1))`。

### 通道相关
`luma_correlation`（c）通过 fotufilm 的 `mixChromaGrain` 决定通道相关：

```
s = sqrt(3 − 2c²) − c
corr = [(2cs + s²)/3] / [c² + (2cs + s²)/3]
```

拿它当 `rho_for_target_correlation(target=corr)` 的目标；`luma_correlation` 为 null 时沿用 0.79。
（`c=0.5 → corr 0.75`，`c=0.6 → corr 0.64`；P0 用 colour_grain=0.5 实测 0.79，同量级。）

### 实施方案（必须预览+导出同改，否则退回"两套实现"）
1. `core/grain_model.py`：加上面四种律 + `granularity_anchor()` + `density_modulation(stock, D)`；`derive()` 接收 stock preset。
2. **把每胶卷的 `mod(D)` 烘焙成 256×1 RGB 查找纹理**，D 归一化到 [0, 2.6] 作 u 坐标。这样以后加律不用改 GLSL。
3. `ui/canvas_viewport.py`：加 `uniform sampler2D u_grain_density_lut` + 上传函数；`paintGL` 与 `render_offscreen` 两处绑定。**保留旧的钟形作为 `u_grain_density_enable==0` 时的回退**，避免纹理没绑上就整片失真。
4. 验收：`tools/grain_stats.py` 需按"当前胶卷"取目标；至少覆盖一支 dyeCloud、一支 silver、一支 reversal，确认 σ(D) 形状各自符合其律；`grain_parity.py` 的 33 项必须仍全绿。

约 180 行，分 `grain_model.py` 与 `canvas_viewport.py`。

---

## 4. 待办二：正片模式下禁用无效控件（小改动、独立）

用户反馈：选中 Instax / 任何正片时，界面上大量控件**能调但实际不生效**。建议规则（一致、可维护）：

> 选中任何正片 → 相纸锁定为"无"并禁用相纸区；印相/化学区（扩印 CMY、扩印曝光、预闪、DIR、显影形态、显影疲劳）整块禁用并挂悬停说明；**胶卷选择保持可用**（正片本身就是胶卷）。

现状：导出路径已为正片强制 `cmy=0 / print_exp=1 / preflash=0`，DIR 也已被 `dir_status()` 置灰，但 UI 上其余控件仍是可点的——界面与行为不一致。

---

## 5. 踩过的坑（别重犯）

| 坑 | 结论 |
|---|---|
| **PSD 斜率不可作外观门禁** | 估计依赖补丁尺寸（192 px 读 −2.01，384 px 读 −1.74），且单一斜率分不清"白噪声+陡拐点"与"三个宽高斯之和"。按它调参会得到统计达标、观感明显偏粗的结果。现门禁是**自相关半宽**（参考 1.81 px）。 |
| `apply_grain_to_density` 的输入是**光学密度** | 传透射率会让曲线整体反向（密度越大颗粒越少）。 |
| 粒子模型在 D ≥ Dmax 完全饱和 | D=2.2 与 2.6 输出**逐位相同**，是钳位假象。拟合只用 D ≤ 1.8，高处用物理 Dmax=2.24 的显式滚降。 |
| fotufilm 缺省值不是 0 | `grainDensityLaw` 缺省 = 黑白 1 / 反转 3 / 其余 0；`2` 是 legacy 律。转换器写死默认 2 曾害 27/33 支。 |
| 曲线可带**第二速度族群** `secondary` | `CharacteristicCurve.density` 要加上它；漏掉会让曲线形状明显不同。 |
| 反转片曲线与源**负相关**是正确的 | 源存"显影后"密度（上升），档案必须反转（下降），corr ≈ −0.98…−0.999、增益 ≈ −1。 |
| 反转片有**中灰校准的 logE 平移** | Instax 为 +0.14；比对时必须允许平移，否则误判为错误。 |
| Instax 是**反转性**介质 | 当相纸用会让 LUT 退化成常数（实测 min=max=mean=0.801）→ 整幅白图。必须当正片胶卷 + 相纸"无"。 |
| 档案 JSON 可无损重写 | `json.dumps(doc, indent=4, ensure_ascii=False)` 对全部 61 个文件逐字节一致，可安全批量改写。 |
| `get_film_grain_preset()` 只取固定键 | 新增字段若不在此处透传，写进档案也到不了渲染层。 |
| 引擎目录是**硬编码清单 + 动态发现** | 胶卷/相纸都可能被过滤掉（`info.support` / `stage`）。健康指标：**61 个档案 0 遗漏**（现 54 胶卷 + 9 相纸）。 |

---

## 6. 关键文件

```
core/colorspace.py          导出色彩空间（解析器 + numba 内核）
core/grain_model.py         颗粒唯一真源（八度、密度响应、rho 反解、CPU 合成）
core/app_core.py            get_film_grain_preset / get_film_stocks / get_paper_stocks / export_image
ui/canvas_viewport.py       着色器（颗粒块 + uniform 绑定两处）
tools/validate_fotufilm_port.py   759 项档案校验（含解析曲线复算）
tools/sync_fotufilm_grain.py      档案字段同步/修复
tools/grain_reference.py          P0 参考生成 + 钟形拟合
tools/grain_stats.py              统计测量（σ(D)、自相关宽度、峰度、PSD）
tools/grain_parity.py             颗粒一致性验收 33 项
tools/grain_params.json           P0 标定目标与参数
tools/grain_visual.py             生成 docs/grain-ab.png 对照图
docs/grain-rework-plan.md         总计划（含各阶段结论、取消理由）
docs/grain-calibration.md         P0 标定报告 + 判据复盘
resources/profiles/PROFILES_MANIFEST.md   档案归属清单 + 字段语义
```

---

## 7. 新会话开场白（可直接复制）

> 读 `docs/HANDOFF.md` 与 `docs/grain-rework-plan.md`，然后做第 3 节「每胶卷密度律接线」：
> 把四种密度律和每胶卷 `luma_correlation` 接进 `grain_model` 与着色器，用 256×1 查找纹理，
> 保留旧钟形作为未绑定纹理时的回退。做完跑 `tools/grain_parity.py`、`tools/grain_stats.py`、
> `tools/validate_fotufilm_port.py`，全绿后更新 CHANGELOG 与版本号。
