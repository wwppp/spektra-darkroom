"""core/i18n.py — 界面文案翻译（中文原文作 msgid）。

设计取舍（见 docs/i18n-plan.md）：**用中文原文当键**，不另起 `exposure.label` 这种键名。
- 漏翻时回落原文，界面最多是中文，不会崩、不会露出 key；
- 因此**一旦某条文案进了 en_US.json，就不要再改它的中文措辞**（改了那条翻译失效）。

用法::

    from core.i18n import t, t_plural
    label.setText(t("曝光"))
    self.set_app_status(t("已导入 {n} 张", n=count))
    self.set_app_status(t_plural("{n} 张底片", count))

语言文件：``resources/i18n/<code>.json``，值是字符串，或（需要单复数时）长度 2 的数组
``["{n} photo", "{n} photos"]``。``index.json`` 决定首选项里能选哪些语言。
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

SOURCE_LANGUAGE = "zh_CN"
DEFAULT_LANGUAGE = SOURCE_LANGUAGE
ENV_VAR = "SPEKTRA_LANG"

_language = None                  # 当前语言代码，None = 还没初始化
_table = {}                       # msgid -> str | [one, many]
_available = {}                   # code -> 显示名
_listeners = []                   # 语言变更回调（热切换用，目前只做重启生效）
_missing_logged = set()


# ---------------------------------------------------------------------------
# 资源路径
# ---------------------------------------------------------------------------
def _i18n_dir():
    from core.path_utils import get_resource_dir
    return os.path.join(get_resource_dir(), "i18n")


def available_languages():
    """``{code: 显示名}``。读不到 index.json 时至少能返回源语言。"""
    _load_index()
    return dict(_available)


def _load_index():
    global _available
    if _available:
        return
    _available = {SOURCE_LANGUAGE: "简体中文"}
    try:
        path = os.path.join(_i18n_dir(), "index.json")
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and data:
            _available = {str(k): str(v) for k, v in data.items()}
    except Exception as ex:
        logger.debug(f"i18n index unreadable: {ex}")


# ---------------------------------------------------------------------------
# 语言切换
# ---------------------------------------------------------------------------
def current_language():
    global _language
    if _language is None:
        set_language(_preferred_language())
    return _language


def _preferred_language():
    """环境变量 > 首选项 > 源语言。任何一步失败都退回源语言。"""
    env = (os.environ.get(ENV_VAR) or "").strip()
    if env:
        return env
    try:
        from core import config_manager
        code = (config_manager.get_preference("ui_language", "") or "").strip()
        if code:
            return code
    except Exception:
        pass
    return DEFAULT_LANGUAGE


def set_language(code):
    """切到 ``code``；找不到该文件时回落源语言（并记一次 warn）。"""
    global _language, _table, _missing_logged
    code = (code or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    table = {}
    if code != SOURCE_LANGUAGE:
        path = os.path.join(_i18n_dir(), f"{code}.json")
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                table = {k: v for k, v in data.items() if isinstance(v, (str, list))}
            else:
                logger.warning(f"i18n file {path} is not an object; falling back to source")
        except FileNotFoundError:
            logger.warning(f"i18n file missing for {code}: {path}")
        except Exception as ex:
            logger.warning(f"i18n file unreadable for {code}: {ex}")
    _language = code
    _table = table
    _missing_logged = set()
    for cb in list(_listeners):
        try:
            cb(code)
        except Exception:
            logger.debug("i18n listener failed", exc_info=True)
    return code


def on_language_changed(callback):
    """注册语言变更回调（供将来做热切换）。返回一个注销函数。"""
    _listeners.append(callback)

    def _off():
        try:
            _listeners.remove(callback)
        except ValueError:
            pass
    return _off


# ---------------------------------------------------------------------------
# 取词
# ---------------------------------------------------------------------------
def _lookup(msgid):
    if _language is None:
        current_language()
    value = _table.get(msgid)
    if value is None:
        if _table and msgid not in _missing_logged:
            _missing_logged.add(msgid)
            logger.debug(f"i18n missing: {msgid!r} ({_language})")
        return msgid
    if isinstance(value, list):
        # 不是通过 t_plural 调的：取复数形（[1]），中文源语言不会走到这里
        return value[1] if len(value) > 1 else (value[0] if value else msgid)
    return value or msgid


def _format(text, kwargs):
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        # 译文把占位符写坏了也不能让界面炸：原样返回
        logger.debug(f"i18n format failed for {text!r} with {kwargs!r}")
        return text


def t(msgid, **kwargs):
    """取译文并按需插值。``msgid`` 就是中文原文。"""
    return _format(_lookup(msgid), kwargs)


def label_width(zh_px, other_px):
    """标签列的固定宽度：中文 3-4 字够用，拉丁文写同样的意思要长得多。

    例：``"色温"`` 74 px 够，``"Temperature"`` 要 100 以上，否则被裁成
    "Temperat…"。所有用到固定标签列的地方都走这里。
    """
    return zh_px if current_language() == SOURCE_LANGUAGE else other_px


def t_plural(msgid, n, **kwargs):
    """单复数取词。JSON 里该条写成 ``["{n} photo", "{n} photos"]``。"""
    if _language is None:
        current_language()
    value = _table.get(msgid)
    if isinstance(value, list) and value:
        text = value[0] if abs(n) == 1 or len(value) == 1 else value[-1]
    else:
        text = _lookup(msgid)
    kwargs.setdefault("n", n)
    return _format(text, kwargs)


def reset_for_tests():
    """探针用：下次取词时重新读环境变量与配置文件。"""
    global _language, _table, _available, _missing_logged
    _language = None
    _table = {}
    _available = {}
    _missing_logged = set()
