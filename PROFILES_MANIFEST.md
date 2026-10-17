# Spektra Darkroom 胶卷与相纸配置文件全量归属清单 (Profiles Manifest)

> **文档作用**: 明确区分本项目中哪些配置属于 **Spektrafilm 官方原生配置**，哪些属于 **从 Fotufilm-engine 转换导入的外来配置**。在新会话或任何后续代码维护中，此文件为绝对真理源（Single Source of Truth）。
> **配置目录**: `resources/profiles/`
> **总计文件**: 61 个（原生 28 个 + 转换 33 个）

---

## 快速判断方法（三保险）

1. **查阅 JSON 内部元数据 (`metadata`)**:
   - **原生配置**: `"metadata"."version": "0.3.2"`，`"metadata"."copyright": "Copyright (c) 2026 Andrea Volpato..."`
   - **外来转换配置**: `"metadata"."version": "fotufilm-port-2.0"`，`"metadata"."datasource": "Converted from fotufilm-engine profile 'xxx'..."`
2. **Git 版本控制状态**:
   - `spektrafilm-repo` 初始 commit `3bb2c2d` 中已存在的为原生配置。
   - `resources/profiles/` 中新增的未跟踪文件全部为 Fotufilm 转换导入配置。
3. **查阅本文档下表**。

---

## 一、 Spektrafilm 官方原生配置（共 28 款）

### 1. 原生底片胶卷（20 款）

| 序号 | 文件名 (`resources/profiles/`) | 胶卷显示名称 | 胶卷类型 | 说明/特性 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `fujifilm_provia_100f.json` | Fujifilm Provia 100F | 正片 (Reversal) | 原生反转片，中灰点密度约 0.51 |
| 2 | `fujifilm_velvia_100.json` | Fujifilm Velvia 100 | 正片 (Reversal) | 原生反转片，中灰点密度约 0.51 |
| 3 | `kodak_ektachrome_100.json` | Kodak Ektachrome E100 | 正片 (Reversal) | 原生反转片，中灰点密度约 0.51 |
| 4 | `kodak_kodachrome_64.json` | Kodak Kodachrome 64 | 正片 (Reversal) | 原生反转片，中灰点密度约 0.51 |
| 5 | `fujifilm_c200.json` | Fujifilm C200 | 彩色负片 (Negative) | 经典日系平民卷 |
| 6 | `fujifilm_pro_400h.json` | Fujifilm Pro 400H | 彩色负片 (Negative) | 经典人像负片，青绿空气感 |
| 7 | `fujifilm_xtra_400.json` | Fujifilm Superia X-TRA 400 | 彩色负片 (Negative) | 真实物理光晕强度 0.0055 |
| 8 | `kodak_ektar_100.json` | Kodak Ektar 100 | 彩色负片 (Negative) | 超细腻高饱和风光负片 |
| 9 | `kodak_gold_200.json` | Kodak Gold 200 | 彩色负片 (Negative) | 温暖金黄暖调经典卷 |
| 10 | `kodak_portra_160.json` | Kodak Portra 160 | 彩色负片 (Negative) | 极细颗粒专业人像卷 |
| 11 | `kodak_portra_400.json` | Kodak Portra 400 | 彩色负片 (Negative) | 现代人像黄金标准卷 |
| 12 | `kodak_portra_800.json` | Kodak Portra 800 | 彩色负片 (Negative) | 高速弱光专业人像卷 |
| 13 | `kodak_portra_800_push1.json` | Kodak Portra 800 Push +1 | 彩色负片 (Negative) | 迫冲 1 挡预设 |
| 14 | `kodak_portra_800_push2.json` | Kodak Portra 800 Push +2 | 彩色负片 (Negative) | 迫冲 2 挡预设 |
| 15 | `kodak_ultramax_400.json` | Kodak UltraMax 400 | 彩色负片 (Negative) | 鲜艳随身全能卷 |
| 16 | `kodak_verita_200d.json` | Kodak Vericolor / Verita 200D | 彩色负片 (Negative) | 经典商业纪实卷 |
| 17 | `kodak_vision3_50d.json` | Kodak Vision3 50D 5203 | 电影负片 (Cinema Neg) | 日光型极细颗粒电影卷 |
| 18 | `kodak_vision3_200t.json` | Kodak Vision3 200T 5213 | 电影负片 (Cinema Neg) | 钨丝灯型电影卷 |
| 19 | `kodak_vision3_250d.json` | Kodak Vision3 250D 5207 | 电影负片 (Cinema Neg) | 日光型主力电影卷 |
| 20 | `kodak_vision3_500t.json` | Kodak Vision3 500T 5219 | 电影负片 (Cinema Neg) | 高速夜景电影卷 |

### 2. 原生相纸（8 款）

| 序号 | 文件名 (`resources/profiles/`) | 相纸显示名称 | 相纸类型 |
| :--- | :--- | :--- | :--- |
| 1 | `fujifilm_crystal_archive_typeii.json` | Fujicolor Crystal Archive Type II | 彩色相纸 |
| 2 | `kodak_2383.json` | Kodak Vision Color Print Film 2383 | 电影放映拷贝胶卷（相纸级） |
| 3 | `kodak_2393.json` | Kodak Vision Premier Color Print Film 2393 | 高饱和影院拷贝胶卷 |
| 4 | `kodak_ektacolor_edge.json` | Kodak Ektacolor Edge Paper | 专业扩印相纸 |
| 5 | `kodak_endura_premier.json` | Kodak Professional Endura Premier Paper | 顶级专业人像放大相纸 |
| 6 | `kodak_portra_endura.json` | Kodak Professional Portra Endura Paper | 经典人像相纸 |
| 7 | `kodak_supra_endura.json` | Kodak Professional Supra Endura Paper | 均衡型商业相纸 |
| 8 | `kodak_ultra_endura.json` | Kodak Professional Ultra Endura Paper | 超高反差相纸 |

---

## 二、 从 Fotufilm 转换导入配置（共 33 款）

全部由 `tools/convert_fotufilm_stocks.py` 从 Fotufilm-engine 数据转换而来，包含物理光谱灵敏度、Hurter-Driffield 染料特征曲线及物理光晕模型。

### 1. 转换版反转片/正片（11 款，需中灰校准）

| 序号 | 文件名 (`resources/profiles/`) | 胶卷显示名称 | Fotufilm 原始 ID | 状态与校准要求 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `fujifilm_velvia_50.json` | Fujifilm Velvia 50 | `velvia50` | 重点校准：中灰密度从 0.97 校准至 0.52 |
| 2 | `agfa_agfachrome_ct_50.json` | Agfacolor Agfachrome CT 50 | `agfachrome50` | 校准中灰密度对齐至 0.52 |
| 3 | `agfa_agfachrome_ct_100.json` | Agfacolor Agfachrome CT 100 | `agfachrome100` | 校准中灰密度对齐至 0.52 |
| 4 | `agfa_agfachrome_ct_200.json` | Agfacolor Agfachrome CT 200 | `agfachrome200` | 校准中灰密度对齐至 0.52 |
| 5 | `fujifilm_astia_100f.json` | Fujifilm Astia 100F | `astia100f` | 校准中灰密度对齐至 0.52 |
| 6 | `kodak_kodachrome_25.json` | Kodak Kodachrome 25 | `kodachrome25` | 校准中灰密度对齐至 0.52 |
| 7 | `kodak_kodachrome_200.json` | Kodak Kodachrome 200 | `kodachrome200` | 校准中灰密度对齐至 0.52 |
| 8 | `kodak_aerochrome_1443.json` | Kodak Aerochrome 1443 CIR | `aerochrome1443` | 红外正片，校准中灰 |
| 9 | `color_infrared_experimental.json` | Color Infrared Experimental | `infraredcolor` | 特效红外正片，校准中灰 |
| 10 | `fujifilm_instax_mini.json` | Fujifilm Instax Mini | `instaxmini` | 拍立得正片，校准中灰 |
| 11 | `fujifilm_instax_wide.json` | Fujifilm Instax Wide | `instaxwide` | 拍立得正片，校准中灰 |

### 2. 转换版黑白负片（7 款）

| 序号 | 文件名 (`resources/profiles/`) | 胶卷显示名称 | Fotufilm 原始 ID | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `fujifilm_neopan_acros_100.json` | Fujifilm Neopan Acros 100 | `acros100` | 超极细颗粒黑白负片 |
| 2 | `ilford_delta_3200.json` | Ilford Delta 3200 Professional | `delta3200` | 超高速大颗粒弱光黑白负片 |
| 3 | `ilford_fp4_plus_125.json` | Ilford FP4 Plus 125 | `fp4plus125` | 细腻影调经典全能黑白 |
| 4 | `ilford_hp5_plus_400.json` | Ilford HP5 Plus 400 | `hp5plus400` | 宽容度之王，纪实街拍黑白 |
| 5 | `kodak_double_x_5222.json` | Kodak Eastman Double-X 5222 | `doublex5222` | 好莱坞经典黑白电影底片 |
| 6 | `kodak_tmax_100.json` | Kodak T-Max 100 | `tmax100` | T 颗粒高解像力黑白 |
| 7 | `kodak_tri_x_400.json` | Kodak Professional Tri-X 400 | `trix400` | 传奇黑白纪实胶卷，高反差有力颗粒 |

### 3. 转换版彩色负片、电影负片与特效胶卷（15 款）

| 序号 | 文件名 (`resources/profiles/`) | 胶卷显示名称 | Fotufilm 原始 ID | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cinestill_400d.json` | CineStill 400Dynamic | `cinestill400d` | 无碳层日光电影卷改制 |
| 2 | `cinestill_800t.json` | CineStill 800Tungsten | `cinestill800t` | 传奇夜景红晕钨丝灯电影卷 |
| 3 | `fujifilm_eterna_500t.json` | Fujifilm Eterna 500T | `eterna500` | 柔和低饱和电影负片 |
| 4 | `fujifilm_eterna_vivid_500t.json` | Fujifilm Eterna Vivid 500T | `eternavivid500` | 高反差浓艳电影负片 |
| 5 | `fujifilm_f_64d.json` | Fujifilm F-64D | `f64d` | 极细颗粒日光电影负片 |
| 6 | `fujifilm_pro_160ns.json` | Fujifilm Pro 160NS | `pro160ns` | 专业棚拍人像，清淡肤色 |
| 7 | `fujifilm_reala_500d.json` | Fujifilm Reala 500D | `reala500d` | 第四感色层技术电影负片 |
| 8 | `fujifilm_superia_200.json` | Fujifilm Superia 200 | `superia200` | 富士平民高锐度生活卷 |
| 9 | `harman_phoenix_200.json` | Harman Phoenix 200 | `phoenix200` | 全新原厂实验性高饱和负片 |
| 10 | `kodak_aerocolor_iv_2460.json` | Kodak Aerocolor IV 2460 | `aerocolor2460` | 航空侦察测绘卷，通透高锐 |
| 11 | `kodak_kodacolor_100.json` | Kodak Kodacolor 100 | `kodacolor100` | 浓郁复古 80 年代胶卷 |
| 12 | `lomography_color_tiger_100.json` | Lomography Color Tiger 100 | `lomocolor100` | 鲜艳野性，Lomo 色调 |
| 13 | `lomography_lomochrome_metropolis.json` | Lomography LomoChrome Metropolis | `lomochromemetropolis` | 去饱和冷硬金属调 |
| 14 | `lomography_lomochrome_purple.json` | Lomography LomoChrome Purple XR | `lomochromepurple` | 绿叶变紫梦幻特效负片 |
| 15 | `lomography_lomochrome_turquoise.json` | Lomography LomoChrome Turquoise XR | `lomochrometurquoise` | 蓝绿异色奇幻特效负片 |

---

## 三、 维护与同步准则

1. 如果通过 `tools/convert_fotufilm_stocks.py` 重新生成配置，输出会严格落在上述 33 款中，不会覆盖 28 款原生配置。
2. 任何需要区分原生与转换的代码逻辑，可通过读取 JSON 的 `metadata.version == "fotufilm-port-2.0"` 精准判定。
