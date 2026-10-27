# 交接文档（HANDOFF）

> 用途：把当前工作从旧会话交接给新会话。**新会话只需读这一个文件 + `docs/grain-rework-plan.md`**，不需要旧对话。
> 最后更新：2026-10-27，v0.5.0 / Build 217

---

## 0. 三十秒上手

```bash
cd D:\spektra-darkroom
python tools/validate_fotufilm_port.py --quiet     # 759 项档案校验，应全 PASS
python tools/grain_parity.py                       # 54 项颗粒一致性 + 密度律，应全 PASS
python tools/grain_stats.py                        # P0 钟形 7 行 + 每胶卷密度律 26 行，应全 PASS
python tools/export_colorspace_check.py            # 41 项色彩空间
python tools/test_export_features.py               # 12 项导出格式
python scratch/dir_probe10.py                      # 全功能冒烟，看 "python exceptions: 0"
python scratch/grain_law_probe.py                  # 54 支胶卷逐支烘焙检查，看 "python exceptions: 0"
```

（PowerShell 下 `$env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONPATH="D:\spektra-darkroom"` 先设好；末行 `[exit code: 1]` 是已知的 harness 收尾现象，判定看 `RESULT: PASS`。`export_colorspace_check` 在连续跑重套件后偶尔会因计时行抖动报 FAIL，单独重跑即 41/0。）

**需要真实 GL 上下文的两项**（offscreen 平台创建不了 GL，前面的脚本都不覆盖渲染）：

```bash
python scratch/gpu_grain_law_check.py       # GPU↔CPU 逐律一致性（会短暂开一个窗口）
python scratch/gpu_paint_bind_ast_check.py  # 静态核对 paintGL/render_offscreen 绑定调用
```

**注意**：本机环境下 `QOpenGLWidget` **不会自行创建 GL 上下文**（`_gl_initialized` 恒为 False，画面 100×30），所以 `paintGL` 的运行时行为无法在脚本里验证。`render_offscreen` 可以（同一上下文、同一套绑定代码），因此渲染路径靠它验收；`paintGL` 只做了 AST 静态核对（`gpu_paint_bind_ast_check.py` 确认 `funcs`/`u_1f`/`u_1i`/`_spec` 都在调用行之前定义）。

---

## 1. 工作区状态

- 版本 **0.5.0 / Build 217**（`core/version.py`）
- v0.5.0 已提交（含本文档）；工作树应干净
- 新增未跟踪（应保持）：`core/colorspace.py`、`core/grain_model.py`、`ui/tooltip.py`、`docs/` 下文档
- **`tools/`、`scratch/`、`resources/profiles/PROFILES_MANIFEST.md`、`fotufilm-engine/` 按项目惯例被 `.gitignore` 排除**（`.gitignore` 第 71–88 行"开发与调试工具"），所以这些脚本与清单只在工作区里、不在版本库中。`git clean -xfd` 会把它们删掉——别在有未备份验证脚本时执行它。

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
| **密度律接线** | 四种密度律 + 每胶卷 `luma_correlation` 接进 `grain_model` 与着色器；烘焙 512×1 RGB 表；预览/导出同表 | **0.5.0** |

**已取消**：P3′（异步 CPU 颗粒场）、P4′（CPU 真 MC）。理由见 `docs/grain-rework-plan.md`——实测 4K 颗粒场 1165 ms（预算 300 ms），粒子模型 24 MP 49 s（预算 10 s）；计划里的性能数字是估算、从未测量。

---

## 3. 已完成：每胶卷密度律接线（0.5.0）

### 做了什么

1. **`core/grain_model.py`** 新增密度律段：四种律的方差式、`granularity_anchor_density()`、`correlation_for_luma()`、`bake_density_lut()`、`_sample_density_lut()`；`derive()` 收 `density_preset`，`GrainSpec` 带 `density_lut`。
2. **烘焙成表**：`DENSITY_LUT_SIZE = 512`、`DENSITY_LUT_MAX = 4.0`（`D = −log10(clamp(lin,1e-4,1))` 的上界，坐标无需 clamp）。表内容 = `DENS_GAIN · mod(D) · 滚降`，逐记录三行。
3. **着色器**：`u_grain_density_lut` / `u_grain_density_scale` / `u_grain_density_enable`，`textureLod` 逐记录取样；`enable == 0` 走原 P0 钟形回退。绑定在 `paintGL` 与 `render_offscreen` 两处（纹理单元 2）。
4. **数据链**：`get_film_grain_preset()` 新增 `density_min/max`（由 `data.density_curves` 现算）；无胶卷与未编录档位 `density_law = None` → 继续用钟形（行为不变）。

### 关键决定（别推翻，除非重新测量）

- **归一化在读数密度处**：每支律除以自己在 `anchor` 的值，于是 `response(anchor) == DENS_GAIN == 0.72`。这是与 P0 的衔接点：旧钟形在 D=1.0 处是 0.7157，差 0.6%。**换律只改形状，不改中密度强度**。
- **`luma_correlation → corr = 2c − c²`**，不是 `1 − c²`。后者在 c=1 处得 0，而 fotufilm 对每支黑白都强制 c=1 → 黑白会变成三通道独立（彩色雪花）。两式在 c=0.5 处同为 0.75，只有端点能区分。
- **每支自己的 Dmax**（曲线跨度 1.42–3.96），不再统一 2.24；越过各自 Dmax 后滚降到 0。
- **表而不是 GLSL 公式**：加律只改 Python。

### 实测外形（`tools/grain_stats.py` 打印）

| 律 | 支数 | 峰位 D | 峰值（=×锚点） | 方向 |
|---|---|---|---|---|
| dyeCloud | 30 | 0.19 | 1.04（1.45×） | 趾部上方见峰后缓降 |
| silver | 7 | 1.96 | 1.54（2.14×） | 单调上升 |
| dyeCloudReversal | 15 | 视 Dmax（如 3.05） | 最高 1.75（2.43×） | 上升后饱和 |
| dyeCloudSelwyn | 1 | 1.57 | 0.90（1.24×） | `√D` |

**GPU↔CPU 一致性**（`scratch/gpu_grain_law_check.py`，真实 `render_offscreen`）：9 组律/钟形幅度比全部 ≤2.8%；D=1.0（锚点）处四律均为 1.00。

---

## 4. 待办：正片模式下禁用无效控件（小改动、独立）

用户反馈：选中 Instax / 任何正片时，界面上大量控件**能调但实际不生效**。建议规则（一致、可维护）：

> 选中任何正片 → 相纸锁定为"无"并禁用相纸区；印相/化学区（扩印 CMY、扩印曝光、预闪、DIR、显影形态、显影疲劳）整块禁用并挂悬停说明；**胶卷选择保持可用**（正片本身就是胶卷）。

现状：导出路径已为正片强制 `cmy=0 / print_exp=1 / preflash=0`，DIR 也已被 `dir_status()` 置灰，但 UI 上其余控件仍是可点的——界面与行为不一致。

---

## 5. 踩过的坑（别重犯）

| 坑 | 结论 |
|---|---|
| **PSD 斜率不可作外观门禁** | 估计依赖补丁尺寸（192 px 读 −2.01，384 px 读 −1.74），且单一斜率分不清"白噪声+陡拐点"与"三个宽高斯之和"。按它调参会得到统计达标、观感明显偏粗的结果。现门禁是**自相关半宽**（参考 1.81 px）。 |
| `apply_grain_to_density` 的输入是**光学密度** | 传透射率会让曲线整体反向（密度越大颗粒越少）。 |
| 粒子模型在 D ≥ Dmax 完全饱和 | D=2.2 与 2.6 输出**逐位相同**，是钳位假象。拟合只用 D ≤ 1.8，高处用物理 Dmax 的显式滚降。 |
| fotufilm 缺省值不是 0 | `grainDensityLaw` 缺省 = 黑白 1 / 反转 3 / 其余 0；`2` 是 legacy 律。转换器写死默认 2 曾害 27/33 支。 |
| 曲线可带**第二速度族群** `secondary` | `CharacteristicCurve.density` 要加上它；漏掉会让曲线形状明显不同。 |
| 反转片曲线与源**负相关**是正确的 | 源存"显影后"密度（上升），档案必须反转（下降），corr ≈ −0.98…−0.999、增益 ≈ −1。 |
| 反转片有**中灰校准的 logE 平移** | Instax 为 +0.14；比对时必须允许平移，否则误判为错误。 |
| Instax 是**反转性**介质 | 当相纸用会让 LUT 退化成常数（实测 min=max=mean=0.801）→ 整幅白图。必须当正片胶卷 + 相纸"无"。 |
| 档案 JSON 可无损重写 | `json.dumps(doc, indent=4, ensure_ascii=False)` 对全部 61 个文件逐字节一致，可安全批量改写。 |
| `get_film_grain_preset()` 只取固定键 | 新增字段若不在此处透传，写进档案也到不了渲染层。 |
| 引擎目录是**硬编码清单 + 动态发现** | 胶卷/相纸都可能被过滤掉（`info.support` / `stage`）。健康指标：**61 个档案 0 遗漏**（现 54 胶卷 + 9 相纸）。 |
| **着色器有 `u_is_bypass` 提前返回** | `if (u_is_bypass == 1) { … return; }` 在颗粒块**之前**。探针不设 `film_stock`（或设为 `"none"`）时整条胶片/颗粒管线被跳过，画面等于纯 sRGB 编码——曾让我误判"颗粒没接上"。任何用 `render_offscreen` 测颗粒的脚本都必须写一个真实 `film_stock`。 |
| **`render_offscreen` 在 ≤512 px 时强制关颗粒** | `is_thumbnail` 条件里有 `max(w,h) <= 512`。测颗粒必须 ≥ 640 px 并且用 `bit_depth=16/32` 读回（8-bit 下细颗粒会被量化掉）。 |
| `exposure` 不是曝光参数名 | 是 `exposure_ev`（+`base_ev`）。写错键不会报错，只会静默无效。 |
| **本机 `paintGL` 不会自己跑起来** | `QOpenGLWidget` 在这个会话环境里拿不到上下文（`_gl_initialized` 恒 False、画布 100×30），所以"预览路径"只能静态核对 + 靠 `render_offscreen` 代跑；别指望 `grabFramebuffer()` 能验证预览。 |
| **`1 − c²` 与 `2c − c²` 只在端点可分** | 见 §3；用 c=0.5 做单点校验会把错误公式判为正确。 |

---

## 6. 关键文件

```
core/colorspace.py          导出色彩空间（解析器 + numba 内核）
core/grain_model.py         颗粒唯一真源（八度、四种密度律、LUT 烘焙、rho 反解、CPU 合成）
core/app_core.py            get_film_grain_preset / _profile_density_range / get_film_stocks / export_image
ui/canvas_viewport.py       着色器（颗粒块 + 密度律纹理 + uniform 绑定两处）
tools/validate_fotufilm_port.py   759 项档案校验（含解析曲线复算）
tools/sync_fotufilm_grain.py      档案字段同步/修复
tools/grain_reference.py          P0 参考生成 + 钟形拟合
tools/grain_stats.py              统计测量 + 四种律的逐律形状门禁
tools/grain_parity.py             颗粒一致性 + 密度律验收 54 项
tools/grain_params.json           P0 标定目标与参数
tools/grain_visual.py             生成 docs/grain-ab.png 对照图（仍只覆盖 P0 钟形）
docs/grain-rework-plan.md         总计划（含各阶段结论、取消理由、P5 密度律结论）
docs/grain-calibration.md         P0 标定报告 + 判据复盘
resources/profiles/PROFILES_MANIFEST.md   档案归属清单 + §3.5 渲染层消费方式
scratch/grain_law_probe.py        54 支胶卷逐支烘焙检查（引擎路径）
scratch/gpu_grain_law_check.py    真实 GL 下的 GPU↔CPU 逐律一致性
scratch/gpu_paint_bind_ast_check.py  paintGL / render_offscreen 绑定调用的静态核对
```

---

## 7. 新会话开场白（可直接复制）

> 读 `docs/HANDOFF.md` 与 `docs/grain-rework-plan.md`。
> 第一件事是跑第 0 节的四套脚本，确认 759 / 54 / 41 / 12 全绿、冒烟 0 异常，再动代码。
> 然后做第 4 节「正片模式下禁用无效控件」：选中任何正片时相纸锁定"无"并禁用相纸区、
> 印相/化学区整块禁用并挂悬停说明，**胶卷选择保持可用**。
