#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich Spektra native profiles with Fotufilm grain data,
polish all converted profile names, badges, and descriptions,
and eliminate redundant *_fotu duplicate profiles.
"""

import os
import sys
import json
import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIRS = [
    os.path.join(ROOT_DIR, 'resources', 'profiles'),
    os.path.join(ROOT_DIR, 'spektrafilm-repo', 'src', 'spektrafilm', 'data', 'profiles')
]
FOTU_STOCKS_DIR = os.path.join(ROOT_DIR, 'fotufilm-engine', 'Sources', 'FotufilmCore', 'Stocks')

# 1. Mapping from Spektra native profiles to corresponding Fotufilm source stock
BUILTIN_TO_FOTU = {
    'kodak_portra_400': ('portra400', 1.0, 1.0),
    'kodak_portra_160': ('portra400', 0.90, 0.75),   # finer, cleaner
    'kodak_portra_800': ('portra400', 1.30, 1.45),   # coarser
    'kodak_portra_800_push1': ('portra400', 1.40, 1.65),
    'kodak_portra_800_push2': ('portra400', 1.50, 1.90),
    'kodak_gold_200': ('gold200', 1.0, 1.0),
    'kodak_ektar_100': ('ektar100', 1.0, 1.0),
    'kodak_ultramax_400': ('ultramax400', 1.0, 1.0),
    'fujifilm_c200': ('c200', 1.0, 1.0),
    'fujifilm_pro_400h': ('pro400h', 1.0, 1.0),
    'fujifilm_xtra_400': ('superia400', 1.0, 1.0),
    'kodak_vision3_250d': ('vision250d', 1.0, 1.0),
    'kodak_vision3_500t': ('vision500t', 1.0, 1.0),
    'kodak_vision3_200t': ('vision500t', 0.90, 0.85),
    'kodak_vision3_50d': ('vision250d', 0.80, 0.65),
    'kodak_verita_200d': ('verita200d', 1.0, 1.0),
    'fujifilm_velvia_100': ('velvia50', 1.0, 1.0),
    'fujifilm_provia_100f': ('provia100f', 1.0, 1.0),
    'kodak_ektachrome_100': ('ektachromee100', 1.0, 1.0),
    'kodak_kodachrome_64': ('kodachrome64', 1.0, 1.0),
}

# 2. Curated Polish Metadata for all non-builtin converted profiles
CONVERTED_POLISH_META = {
    'agfa_agfachrome_ct_100': {
        'name': 'Agfa Agfachrome CT 100',
        'badge': '欧系正片',
        'desc': '爱克发传奇彩色反转片，独特的欧系浓郁红黄色调，通透而富有油画质感。'
    },
    'agfa_agfachrome_ct_200': {
        'name': 'Agfa Agfachrome CT 200',
        'badge': '中速反转',
        'desc': '中速反转片，颗粒更具张力与存在感，阴天与弱光下呈现深沉富有力量感的冷暖对比。'
    },
    'agfa_agfachrome_ct_50': {
        'name': 'Agfa Agfachrome CT 50',
        'badge': '极细微粒',
        'desc': '低感高解像反转片，极致纯净细腻的暗部与鲜明柔滑的花草风光表现。'
    },
    'cinestill_400d': {
        'name': 'CineStill 400D',
        'badge': '暖调电影',
        'desc': '日光型电影底片去碳层改良版，柔和自然反差与标志性的高光微暖光晕。'
    },
    'cinestill_800t': {
        'name': 'CineStill 800T',
        'badge': '夜景光晕',
        'desc': '好莱坞钨丝灯电影底片去碳层，夜景霓虹灯下极具辨识度的迷人红橙光晕与冷调蓝绿暗部。'
    },
    'fujifilm_velvia_50': {
        'name': 'Fujifilm Velvia 50',
        'badge': '极致风光',
        'desc': '一代风光神卷，超高色彩饱和度、深沉黑阶与极高解像力，大自然绚丽色彩的终极呈现。'
    },
    'fujifilm_astia_100f': {
        'name': 'Fujifilm Astia 100F',
        'badge': '人像正片',
        'desc': '专为人像与时尚设计的反转片，柔和雅致的高光反差与极其粉嫩通透的亚洲肤色过渡。'
    },
    'fujifilm_eterna_500t': {
        'name': 'Fujifilm Eterna 500T',
        'badge': '富士电影绝唱',
        'desc': '富士经典高速钨丝灯电影底片，温润沉稳的绿色调与极具包容度的平滑宽容度。'
    },
    'fujifilm_eterna_vivid_500t': {
        'name': 'Fujifilm Eterna Vivid 500T',
        'badge': '高饱和电影',
        'desc': '高饱和度高反差电影底片，原色鲜明立体，为画面注入充满戏剧张力的现代电影质感。'
    },
    'fujifilm_f_64d': {
        'name': 'Fujifilm F-64D',
        'badge': '低感电影卷',
        'desc': '超低感光度日光电影底片，几乎不可见的超微颗粒与极其纯净清透的高光还原。'
    },
    'fujifilm_reala_500d': {
        'name': 'Fujifilm Reala 500D',
        'badge': '四感光层',
        'desc': '搭载富士专利第4感光层的高速电影卷，完美纠正荧光灯偏色，呈现真实生动的丰富色彩。'
    },
    'fujifilm_superia_200': {
        'name': 'Fujifilm Superia 200',
        'badge': '日常民用',
        'desc': '富士普及型生活彩色负片，清丽明快的色彩与轻快自然的绿意，扫街与日常记录经典。'
    },
    'fujifilm_pro_160ns': {
        'name': 'Fujifilm PRO 160NS',
        'badge': '商业人像',
        'desc': '专业影棚人像胶卷，低对比度、宽广动态范围与极其柔滑细腻的高级中性灰阶。'
    },
    'fujifilm_neopan_acros_100': {
        'name': 'Fujifilm Neopan Acros 100 II',
        'badge': '极微晶黑白',
        'desc': 'Super Fine-Σ 专利微晶技术，全球顶级解像力的黑白底片，超强倒易律与锐利灰阶。'
    },
    'fujifilm_instax_mini': {
        'name': 'Fujifilm Instax Mini',
        'badge': '拍立得即显',
        'desc': '富士一次成像相纸，高光微暖、暗部偏蓝的标志性即时物理显影胶片风味。'
    },
    'fujifilm_instax_wide': {
        'name': 'Fujifilm Instax Wide',
        'badge': '宽幅拍立得',
        'desc': '开阔画幅拍立得相纸，柔和自然的显影微粒与生活记录中充满回忆感的复古色调。'
    },
    'kodak_kodacolor_100': {
        'name': 'Kodak Kodacolor 100',
        'badge': '复古彩负',
        'desc': '柯达民用彩色负片鼻祖风格，温暖黄褐基调、浓厚复古胶片感与醇厚的历史年代感。'
    },
    'kodak_double_x_5222': {
        'name': 'Kodak Double-X 5222',
        'badge': '黑白电影传奇',
        'desc': '《辛德勒的名单》《奥本海默》黑白同款底片，戏剧化反差、经典银盐微粒与深沉暗调。'
    },
    'kodak_tmax_100': {
        'name': 'Kodak T-Max 100',
        'badge': 'T-Grain 晶体',
        'desc': 'T-Grain 扁平微晶黑白负片，无可挑剔的极高解像力与近乎察觉不到的极细微粒。'
    },
    'kodak_tri_x_400': {
        'name': 'Kodak Tri-X 400',
        'badge': '纪实黑白圣经',
        'desc': '玛格南摄影师永恒之选，粗粝强劲的银盐质感、强烈明暗反差与充满力量的人文张力。'
    },
    'kodak_aerochrome_1443': {
        'name': 'Kodak Aerochrome 1443',
        'badge': '红外假色',
        'desc': '传奇航测红外反转片，将植物叶绿素渲染为惊艳的艳粉与深红，超现实异次元色彩奇观。'
    },
    'kodak_aerocolor_iv_2460': {
        'name': 'Kodak Aerocolor IV 2460',
        'badge': '航拍彩色',
        'desc': '航空摄影高解像彩色负片，透明超薄无遮罩片基，鲜艳通透的原色与极低阴影颗粒。'
    },
    'kodak_kodachrome_25': {
        'name': 'Kodak Kodachrome 25',
        'badge': '百年染印',
        'desc': '人类历史上最细腻的彩色反转片，K-14 外偶联染料工艺，油画般厚重纯正且永不褪色。'
    },
    'kodak_kodachrome_200': {
        'name': 'Kodak Kodachrome 200',
        'badge': '高速染印',
        'desc': '高速外偶联反转片，浓郁醇厚的红色与温暖深沉的中间调，国家地理经典纪实色彩。'
    },
    'ilford_hp5_plus_400': {
        'name': 'Ilford HP5 Plus 400',
        'badge': '全能黑白',
        'desc': '英国伊尔福传世之作，无与伦比的高容错宽容度、温润饱满的灰阶过渡与街头纪实质感。'
    },
    'ilford_fp4_plus_125': {
        'name': 'Ilford FP4 Plus 125',
        'badge': '细腻灰阶',
        'desc': '经典中速黑白负片，极细致颗粒与长影调中间灰过渡，静物、人像与风光的长青树。'
    },
    'ilford_delta_3200': {
        'name': 'Ilford Delta 3200',
        'badge': '弱光巨兽',
        'desc': '超高感光度黑白底片，Core-Shell 核心晶体技术，粗犷迷人的胶片噪点与夜色电影感。'
    },
    'harman_phoenix_200': {
        'name': 'Harman Phoenix 200',
        'badge': '实验新星',
        'desc': '英国 Harman 全新独立自研彩色负片，无橙色遮罩、高反差与浓烈的实验性复古暖调。'
    },
    'lomography_color_tiger_100': {
        'name': 'Lomography Color Tiger 100',
        'badge': '狂野日光',
        'desc': 'Lomography 经典彩色日光卷，高饱和、高反差，阳光下尽显随性洒脱的胶片冲劲。'
    },
    'lomography_lomochrome_metropolis': {
        'name': 'Lomography Metropolis',
        'badge': '冷调都市',
        'desc': '专为现代钢铁都市设计，低饱和度、硬朗对比度与独特的冷萃暗部电影影调。'
    },
    'lomography_lomochrome_purple': {
        'name': 'Lomography Purple',
        'badge': '迷幻假色',
        'desc': '无需滤镜的奇幻变色胶片，将日常绿色巧妙转变为迷人的粉紫与靛青梦幻景象。'
    },
    'lomography_lomochrome_turquoise': {
        'name': 'Lomography Turquoise',
        'badge': '松石绿异色',
        'desc': '极富实验性的变色胶片，暖调转化为深邃钴蓝，蓝天绿地则呈现灿烂松石绿与金黄。'
    },
    'color_infrared_experimental': {
        'name': 'Color Infrared (红外假色)',
        'badge': '实验红外',
        'desc': '实验性红外光谱胶片模拟，突破人眼可见光谱边界，创造超现实异色梦境。'
    }
}

def extract_fotu_grain(raw_id, size_mult=1.0, strength_mult=1.0):
    p = os.path.join(FOTU_STOCKS_DIR, f"{raw_id}.json")
    if not os.path.exists(p):
        return None
    with open(p, 'r', encoding='utf-8') as fp:
        raw = json.load(fp)
    
    base_size = float(raw.get('grainSizeMM', 0.005))
    base_strength = float(raw.get('grainStrength', 0.010))
    
    return {
        'crystal_grain': {
            'population': raw.get('crystalGrainPopulation'),
            'size_mm': round(base_size * size_mult, 5),
            'strength': round(base_strength * strength_mult, 5),
            'density_profile': raw.get('grainDensityProfile'),
            'layer_weights': raw.get('grainLayerWeights', [0.70, 1.0, 1.35]),
            'density_law': raw.get('grainDensityLaw', 2),
        },
        'halation': {
            'look_scale': raw.get('halationLookScale', 1.0),
            'strength': raw.get('halationStrength', 0.0),
            'return_matrix': raw.get('halationReturnMatrix'),
            'profile': raw.get('halationProfile')
        },
        'couplers': {
            'coupler_diffusion_mm': raw.get('couplerDiffusionMM'),
            'adjacency_radius_mm': raw.get('adjacencyRadiusMM'),
            'adjacency_strength': raw.get('adjacencyStrength', 0.0),
            'coupler_inhibition': raw.get('couplerInhibition'),
        }
    }

def main():
    print("--- 1. Enriching Native Built-in Profiles with Fotufilm Grain ---")
    enriched_count = 0
    for stock_slug, (fotu_id, s_mult, str_mult) in BUILTIN_TO_FOTU.items():
        ext = extract_fotu_grain(fotu_id, s_mult, str_mult)
        if not ext:
            print(f"Warning: Fotufilm stock {fotu_id} not found for {stock_slug}")
            continue

        for p_dir in PROFILES_DIRS:
            target_file = os.path.join(p_dir, f"{stock_slug}.json")
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                
                # Inject physical_extensions
                if 'data' in data and isinstance(data['data'], dict):
                    data['data']['physical_extensions'] = ext
                else:
                    data['physical_extensions'] = ext
                
                with open(target_file, 'w', encoding='utf-8') as fp:
                    json.dump(data, fp, indent=4, ensure_ascii=False)
                
        print(f"Enriched {stock_slug} <- {fotu_id} (size: {ext['crystal_grain']['size_mm']}mm, strength: {ext['crystal_grain']['strength']})")
        enriched_count += 1

    print(f"\nTotal native profiles enriched with physical grain: {enriched_count}")

    print("\n--- 2. Polishing Names, Badges, and Descriptions for Converted Profiles ---")
    polished_count = 0
    for stock_slug, meta in CONVERTED_POLISH_META.items():
        for p_dir in PROFILES_DIRS:
            target_file = os.path.join(p_dir, f"{stock_slug}.json")
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                
                # Update top-level and info
                data['name'] = meta['name']
                data['desc'] = meta['desc']
                data['badge'] = meta['badge']
                
                if 'info' in data and isinstance(data['info'], dict):
                    data['info']['name'] = meta['name']
                    data['info']['desc'] = meta['desc']
                    data['info']['badge'] = meta['badge']
                
                with open(target_file, 'w', encoding='utf-8') as fp:
                    json.dump(data, fp, indent=4, ensure_ascii=False)
                    
        print(f"Polished: {stock_slug} -> '{meta['name']}' [{meta['badge']}]")
        polished_count += 1

    print(f"\nTotal converted profiles polished: {polished_count}")

    print("\n--- 3. Removing Redundant *_fotu Duplicate Profiles ---")
    deleted_count = 0
    fotu_duplicates = []
    for p_dir in PROFILES_DIRS:
        for f in glob.glob(os.path.join(p_dir, "*_fotu.json")):
            os.remove(f)
            fotu_duplicates.append(os.path.basename(f))
            deleted_count += 1
    
    # Also clean filter databases of _fotu entries
    filter_files = [
        os.path.join(ROOT_DIR, 'resources', 'neutral_print_filters.json'),
        os.path.join(ROOT_DIR, 'spektrafilm-repo', 'src', 'spektrafilm', 'data', 'filters', 'neutral_print_filters.json')
    ]
    for ff in filter_files:
        if os.path.exists(ff):
            with open(ff, 'r', encoding='utf-8') as fp:
                filt_db = json.load(fp)
            changed = False
            for p_name, ill_map in filt_db.items():
                if isinstance(ill_map, dict):
                    for ill, stock_map in ill_map.items():
                        if isinstance(stock_map, dict):
                            to_del = [k for k in stock_map if k.endswith('_fotu')]
                            for k in to_del:
                                del stock_map[k]
                                changed = True
            if changed:
                with open(ff, 'w', encoding='utf-8') as fp:
                    json.dump(filt_db, fp, indent=4, ensure_ascii=False)

    print(f"Removed {len(set(fotu_duplicates))} unique redundant _fotu profiles.")
    print("All tasks finished successfully!")

if __name__ == '__main__':
    main()
