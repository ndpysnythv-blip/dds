#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kora Zola - AI 点单智能服务
=================================
同时承担：
  1) 前端静态页面的 HTTP 服务（index.html, customer.html, ...）
  2) /api/ai/chat 自然语言理解接口
     - 首选：调用用户配置的大语言模型（OpenAI / DeepSeek / 通义 / 智谱 等 OpenAI 兼容接口）
     - 降级：内置的本地规则解析引擎（不依赖网络也能用）

运行：
    # 首次运行自动创建 config.json（可手工编辑或在前端设置中修改）
    python ai-server.py

    # 指定端口
    python ai-server.py --port 8000

    # 绑定到所有网卡（允许局域网访问，例如手机扫码点单）
    python ai-server.py --host 0.0.0.0 --port 8000
"""

import os
import re
import sys
import json
import time
import argparse
import datetime
import threading
import urllib.request
import urllib.parse
import urllib.error
import ssl
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

# ============================================================
# 配置管理
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.json')

DEFAULT_CONFIG = {
    "ai": {
        # 是否启用大模型（false 时强制走本地规则）
        "enabled": True,
        # OpenAI 兼容的 API Base URL（填写哪家服务商提供的都可以）
        # 常用：
        #   OpenAI:        https://api.openai.com/v1
        #   DeepSeek:      https://api.deepseek.com/v1
        #   智谱(ZHIPU):   https://open.bigmodel.cn/api/paas/v4
        #   通义(阿里):    https://dashscope.aliyuncs.com/compatible-mode/v1
        #   硅基流动:      https://api.siliconflow.cn/v1
        #   月之暗面:      https://api.moonshot.cn/v1
        #   本地 Ollama:   http://127.0.0.1:11434/v1
        "apiBase": "https://api.deepseek.com/v1",
        # 在服务商获取的 API Key
        "apiKey": "",
        # 使用的模型名（请根据上面所选服务商填写）
        # 例如：deepseek-chat / gpt-4o-mini / glm-4-flash / qwen-plus
        "model": "deepseek-chat",
        # 单次请求温度（0=更确定，1=更随机）
        "temperature": 0.2,
        # 单次超时秒数
        "timeout": 15
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000
    }
}


def load_config():
    """读取配置；不存在或损坏则自动回写一份默认配置。"""
    if not os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('[WARN] 无法写入 config.json:', e)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # 合并缺省字段，向后兼容
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        _deep_merge(merged, cfg)
        return merged
    except Exception as e:
        print('[WARN] 读取 config.json 失败，使用默认配置：', e)
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print('[WARN] 保存 config.json 失败:', e)
        return False


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# 全局配置（运行期间可被 /api/config PUT 修改）
_config_lock = threading.Lock()
CONFIG = load_config()


# ============================================================
# 工具函数
# ============================================================

def log(level, msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    color = {'INFO': '\033[34m', 'OK': '\033[32m', 'WARN': '\033[33m', 'ERR': '\033[31m'}.get(level, '')
    reset = '\033[0m'
    print(f'{color}[{ts}] [{level}] {msg}{reset}')


# 中文数字映射（用于本地降级解析）
_CN_NUM_MAP = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '俩': 2,
    '三': 3, '四': 4, '五': 5, '六': 6, '七': 7,
    '八': 8, '九': 9, '十': 10, '百': 100
}


def cn_to_int(s):
    """非常轻量的中文数字解析；只覆盖点餐口语中常见的 1-99。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # 纯阿拉伯数字
    if re.fullmatch(r'\d+', s):
        try:
            return int(s)
        except Exception:
            return None
    try:
        # 十 / 十一 / 二十 / 二十三
        if s == '十':
            return 10
        if s.startswith('十') and len(s) == 2:
            tail = _CN_NUM_MAP.get(s[1])
            return 10 + tail if tail else None
        if len(s) == 2 and s[1] == '十':
            head = _CN_NUM_MAP.get(s[0])
            return head * 10 if head else None
        if len(s) == 3 and s[1] == '十':
            head = _CN_NUM_MAP.get(s[0])
            tail = _CN_NUM_MAP.get(s[2])
            if head and tail:
                return head * 10 + tail
        if s in _CN_NUM_MAP:
            return _CN_NUM_MAP[s]
    except Exception:
        pass
    return None


# ============================================================
# 大模型调用
# ============================================================

def call_llm(system_prompt, user_prompt):
    """调用 OpenAI 兼容的 chat.completions 接口；失败时抛出异常。"""
    ai_cfg = CONFIG.get('ai', {})
    if not ai_cfg.get('enabled'):
        raise RuntimeError('LLM disabled by config')
    api_base = (ai_cfg.get('apiBase') or '').rstrip('/')
    api_key = ai_cfg.get('apiKey') or ''
    model = ai_cfg.get('model') or ''
    temperature = float(ai_cfg.get('temperature', 0.2))
    timeout = float(ai_cfg.get('timeout', 15))

    if not api_base or not api_key or not model:
        raise RuntimeError('缺少 apiBase / apiKey / model 配置')

    url = api_base + '/chat/completions'
    payload = json.dumps({
        'model': model,
        'temperature': temperature,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'response_format': {'type': 'json_object'}
    }, ensure_ascii=False).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'User-Agent': 'KoraZola-AI/1.0'
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    ctx = ssl.create_default_context()
    # 允许自签/本地证书
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8', errors='ignore')
        except Exception:
            pass
        raise RuntimeError(f'LLM HTTP {e.code}: {detail[:300]}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'LLM URL error: {e.reason}') from e

    raw = resp.read().decode('utf-8', errors='ignore')
    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError('LLM 返回非 JSON: ' + raw[:200]) from e

    try:
        return data['choices'][0]['message']['content']
    except (KeyError, IndexError) as e:
        raise RuntimeError('LLM 返回结构异常: ' + raw[:200]) from e


# ============================================================
# 系统 Prompt：告诉大模型如何理解点单意图
# ============================================================

SYSTEM_PROMPT = """你是一家名叫 Kora Zola 的咖啡店的点单助手。
你需要将用户的自然语言点单意图，解析为严格的 JSON 对象输出。
只输出 JSON，不输出任何解释性文字、Markdown 或代码块。

【商品数据】
__GOODS_JSON_PLACEHOLDER__

【购物车现状】
__CART_JSON_PLACEHOLDER__

【输出 JSON Schema】
{
  "intent": "order | remove | confirm | recommend | price | menu | remark | cancel | chat | unknown",
  "items": [ {"id": number, "name": string, "qty": number} ],
  "remarks": [ string ],
  "reply": string,
  "openCheckout": boolean
}

【字段说明】
- intent：用户意图分类，必须二选一：
    order    = 想要加购商品（含「来一杯XX、两份XX、再点XX、点XX」等）
    remove   = 想要删除 / 去掉购物车里的某商品（含「去掉XX、不要XX、退掉XX」）
    confirm  = 确认当前购物车进入结账（含「确认下单、就这样、好了、结账、买单」）
    recommend = 请求推荐（含「推荐一下、有什么好、喝点什么」）
    price    = 询问某商品价格
    menu     = 查询某分类 / 全部商品列表
    remark   = 只输入备注信息，不新增商品（如「不加糖、少冰」）
    cancel   = 清空 / 取消当前订单
    chat     = 普通闲聊、问候
    unknown  = 确实无法理解
- items：解析到的具体商品数组。必须使用已给的商品 id；商品名用商品数据里的 name 填充；未识别到商品留空数组。
- remarks：用户提到的备注（无糖、少糖、少冰、去冰、热、常温、打包 等）。
- reply：给用户的中文自然语言回复，语气友好、简洁（2 句以内为佳）。
- openCheckout：如果 intent=confirm 或用户明确要结账，填 true；其他情况 false。

【注意】
1. 商品名使用模糊匹配，口语如「拿铁」可以匹配到「拿铁咖啡」「生椰拿铁」等多个候选时，请选最常见的一个（一般包含该关键字的最短名称）。
2. 「一杯、两杯、两份、三个」等数量请正确解析。
3. 「再来一杯」「再加一份」等需结合购物车判断（购物车若只有一款则默认 +1 同款）。
4. 若提到了商品名，但是加购意图不确定时，仍然按 intent=order 加入，并在 reply 中友好提示用户可以继续点或确认下单。
5. 不要编造商品数据中不存在的商品。
"""


# ============================================================
# 本地降级解析引擎（无网络/无 key 时使用）
# ============================================================

def _normalize(text):
    if text is None:
        return ''
    return re.sub(r'\s+', '', str(text)).lower()


def _match_goods(text, goods):
    """在文本中查找商品；返回 [{id,name,qty}]。
    策略（按优先级）：
      1) 完整商品名（或去空格后的）精确包含在用户文本里 → 取该款
      2) 对每个商品拆出 >=2 字的子词，找用户文本中出现过的最长子词 → 取最长的那一款
    默认一次只返回一个商品（避免"美式咖啡/拿铁咖啡"都被误加）。
    """
    if not goods or not text:
        return []
    text_n = _normalize(text)

    # 1) 先找：用户文本里直接完整包含了某个商品名
    for g in goods:
        name = _normalize(g.get('name', ''))
        if not name:
            continue
        if name in text_n:
            qty = _extract_qty_near(text, g.get('name', '')) or 1
            return [{'id': g['id'], 'name': g['name'], 'qty': qty}]

    # 2) 反向：从每个商品名提取所有 >=2 字的连续子词，找出在用户文本中存在的最长子词
    #    例："拿铁咖啡" → 会生成 "拿铁"、"拿铁咖"、"拿铁咖啡"、"铁咖"、"铁咖啡"、"咖啡" 等
    #    这样用户说"来两杯拿铁"，即使没有完整商品名"拿铁咖啡"，也能命中"拿铁"子词
    best = None  # (g, max_subword_len, qty)
    for g in goods:
        name = _normalize(g.get('name', ''))
        if not name:
            continue
        subwords = set()
        for i in range(len(name)):
            for j in range(i + 2, len(name) + 1):
                sub = name[i:j]
                if len(sub) >= 2 and sub in text_n:
                    subwords.add(sub)
        if subwords:
            longest = max(subwords, key=lambda x: len(x))
            qty = _extract_qty_near(text, longest) or 1
            if best is None or len(longest) > best[1]:
                best = (g, len(longest), qty)

    if best:
        return [{'id': best[0]['id'], 'name': best[0]['name'], 'qty': best[2]}]
    return []


def _extract_qty_near(text, keyword):
    if not keyword:
        return None
    idx = text.find(keyword)
    if idx < 0:
        return None
    # 关键词前后各 10 字
    start = max(0, idx - 10)
    end = min(len(text), idx + len(keyword) + 10)
    window = text[start:end]

    # 阿拉伯数字
    m = re.search(r'(\d+)\s*(杯|份|个|件|份|碗|瓶)?', window)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # 「两/三杯」之类的中文数字
    cn = re.findall(r'([零〇一二两俩三四五六七八九十百]+)\s*(杯|份|个|件|碗|瓶)?', window)
    if cn:
        for c in cn:
            n = cn_to_int(c[0])
            if n:
                return n

    # 「再来一杯 / 再来一份」等
    if re.search(r'再来一|再加一|多一|来一|要一|点一', window):
        return 1

    return None


def _extract_remarks(text):
    patterns = [
        r'(无糖|少少糖|少糖|全糖|半糖|多甜|不甜)',
        r'(去冰|少冰|多冰|正常冰|常温|热|冰|加冰)',
        r'(打包|外带|带走|堂食|店内)',
        r'(加奶|加浓|脱脂|燕麦奶|豆奶|换奶)',
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found


def _extract_quantity(text):
    """从文本里提取所有数量词，按出现位置返回 [(词, 位置, 数值)]"""
    if not text:
        return []
    out = []
    # 阿拉伯数字（1-999），可带"杯/份/个/碗/瓶"等量词
    for m in re.finditer(r'(\d{1,3})\s*(杯|份|个|件|碗|瓶|杯|罐)?', text):
        try:
            out.append((m.group(0), m.start(), int(m.group(1))))
        except Exception:
            pass
    # 中文数字
    cn_map = {
        '一': 1, '二': 2, '两': 2, '俩': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '二十': 20, '三十': 30, '一百': 100
    }
    # 先试"二十/三十"
    for word, num in [('二十', 20), ('三十', 30), ('一百', 100)]:
        for m in re.finditer(re.escape(word), text):
            out.append((word, m.start(), num))
    # 再试单个字
    for ch, num in cn_map.items():
        if len(ch) == 1:
            for m in re.finditer(re.escape(ch), text):
                out.append((ch, m.start(), num))
    # 去重：同一个位置只保留最长的匹配
    out.sort(key=lambda x: (x[1], -len(x[0])))
    seen_pos = set()
    cleaned = []
    for entry in out:
        pos = entry[1]
        # 若该位置已经被更长/先出现的词覆盖，则跳过
        if any(p <= pos <= p + 2 for p in seen_pos):
            continue
        seen_pos.add(pos)
        cleaned.append(entry)
    return cleaned


def _smart_match_items(text, goods, cart_items):
    """在文本中找出所有提到的商品 → [{id, name, qty}]
    支持：'拿铁 2 杯'、'两份提拉米苏'、'拿铁和美式各一杯'、'再来一份'（依赖 cart）
    策略：
      a) 先在文本中找出每个商品名/子词的所有命中位置
      b) 在每个命中位置的前后 [-10, +len(name)+10] 内找数量；没数量时默认 1
      c) 支持"再来一份/再要一个/同样再来" → 默认加购物车里第一项（或最常点的）
    """
    if not goods:
        return []
    tn = _normalize(text)
    hits = []  # [(g, pos_in_text, matched_word)]

    # a) 逐商品命中位置（用完整名先，再子词）
    used_positions_by_g = {}
    for g in goods:
        name_n = _normalize(g.get('name', ''))
        if not name_n:
            continue
        candidates = []  # (word, pos)
        if name_n in tn:
            for m in re.finditer(re.escape(name_n), tn):
                candidates.append((name_n, m.start()))
        if not candidates:
            # 生成 >=2 字子词，找首次命中
            sub_seen = set()
            for i in range(len(name_n)):
                for j in range(i + 2, len(name_n) + 1):
                    sub = name_n[i:j]
                    if sub in sub_seen:
                        continue
                    sub_seen.add(sub)
                    if sub in tn:
                        for m in re.finditer(re.escape(sub), tn):
                            candidates.append((sub, m.start()))
                        break  # 每个商品只取一个子词命中（避免被一个词无限重复）
        for c in candidates:
            hits.append((g, c[1], c[0]))

    # b) 对每个命中，在附近找数量；并按商品 id 聚合，取最大位置/最合理数量
    result_map = {}  # id -> {id, name, qty}
    for g, pos, word in hits:
        qty = 1
        start = max(0, pos - 12)
        end = min(len(tn), pos + len(word) + 12)
        window = tn[start:end]
        nums = _extract_quantity(window)
        if nums:
            # 选最靠近命中词的数量
            best = min(nums, key=lambda n: abs(n[1] - (pos - start)))
            qty = best[2] if best else 1
        # "各一份/每样一份/每样一杯" 等强数量词
        if re.search(r'各(一|1)|每样|每个|每杯', window):
            qty = 1
        existing = result_map.get(g['id'])
        if existing:
            existing['qty'] = max(existing['qty'], qty)
        else:
            result_map[g['id']] = {'id': g['id'], 'name': g['name'], 'qty': qty}

    # c) 上下文感知：用户说"再来一份/再要一杯/同样再来/还是这个" 之类
    if not result_map and re.search(r'(再来一份|再来一杯|再加一份|再加一个|同样的|还是这个|还是老样子|再来|再要|再加|一样的|续杯|同样的来一份)', tn):
        if cart_items:
            # 默认加购物车里第一项；如果只有一项，就加 1 份
            first = cart_items[0]
            result_map[first['id']] = {'id': first['id'], 'name': first['name'], 'qty': 1}

    return list(result_map.values())


def _extract_remarks_smart(text, existing_remarks=None):
    """更全面的备注识别：糖度/冰量/温度/打包堂食/规格偏好等。"""
    tn = _normalize(text)
    found = []
    rules = [
        # 糖度
        (r'(无糖|不要糖|不加糖|去糖|零糖|0糖|低糖|少少糖|三分糖|半糖|少糖|多糖|全糖|加甜|甜一点|不要甜|不甜|很甜)', '糖度'),
        # 冰量/温度
        (r'(去冰|少冰|少少冰|冰多一点|冰多|多冰|正常冰|冰|加冰|热|热的|热饮|常温|温|不冰)', '温度/冰量'),
        # 打包
        (r'(打包|外带|带走|外卖|带走喝|拿回家)', '打包'),
        # 堂食
        (r'(堂食|店内|在这喝|在这吃|店里喝|店里)', '堂食'),
        # 杯型/规格
        (r'(大杯|中杯|小杯|大份|小份|加大|超大|大一点|小一点)', '规格'),
        # 配料/口味修饰
        (r'(加奶|加浓|加倍浓缩|脱脂|燕麦奶|豆奶|换奶|淡奶|不要奶|不要奶泡|去奶泡)', '奶/浓度'),
        # 忌口/过敏提示（轻度）
        (r'(不要咖啡|不要咖啡因|低因|脱因)', '特殊要求'),
    ]
    for pat, _label in rules:
        for m in re.finditer(pat, tn):
            w = m.group(1)
            if w and w not in found:
                found.append(w)
    return found


def _classify_intent(text, items_found, has_remark_only, cart_has_items):
    """基于关键词 + 槽位信息，做更稳健的意图判断。
    返回：'cancel' | 'confirm' | 'remove' | 'price' | 'menu' | 'recommend' | 'chat' | 'order' | 'remark' | 'unknown'
    """
    tn = _normalize(text)

    # 1) 取消 / 清空 优先
    if re.search(r'(取消订单|全部取消|全部不要|清空|重来|重新点|重新下单|不要了|算了|不用了|退掉|cancel all|clear all)', tn):
        return 'cancel'

    # 2) 确认 / 结账（纯确认词 且 没有新增商品词）
    confirm_pat = r'(确认下单|确认一下|确认|结账|买单|去支付|支付|付款|就这样|就这些|就这些吧|好了|够了|checkout|done|ok|是的|对|好的|就这个|就这个吧|就这样|行了)'
    if re.search(confirm_pat, tn) and not items_found:
        return 'confirm'

    # 3) 删除 / 去掉某商品
    # - 文本包含"不要/去掉..."类关键字，且其后跟着商品名 → remove
    # - 如果其后只是备注词（冰、糖）→ 不是 remove，后面让它走 remark 处理
    remove_pat = r'(去掉|不要|删除|移除|退|减去|少一份|少一个|去掉这个|别加|去掉那个|去掉这|去掉那|不要这个|不要那个|no|remove)'
    if re.search(remove_pat, tn):
        if items_found:
            return 'remove'
        # 没说商品名，购物车里有东西 → 也算 remove（稍后外层要求用户说明）
        if cart_has_items and not has_remark_only:
            return 'remove'

    # 4) 价格询问
    if re.search(r'(多少钱|价格|价位|多少元|几元|多少钱一杯|how much|price|多少钱一份)', tn):
        return 'price'

    # 5) 菜单 / 有什么（先于推荐）
    if re.search(r'(有什么(咖啡|饮品|甜品|甜点|小食|蛋糕|点心|东西|喝的|吃的)|菜单|有啥|有什么|还有啥|有哪些)', tn):
        return 'menu'

    # 6) 推荐请求
    if re.search(r'(推荐|招牌|热门|什么好|好喝|好吃|喝点什么|吃点什么|来点什么|suggest|recommend|popular|不知道点什么|给点建议|给建议)', tn):
        return 'recommend'

    # 7) 加购商品（有明确商品命中）
    if items_found:
        return 'order'

    # 8) 纯备注（只有备注词，没有其他内容）
    if has_remark_only:
        return 'remark'

    # 9) 问候/闲聊（文本较短且像问候）
    if len(text) <= 12 and re.search(r'(你好|您好|在吗|哈喽|嗨|你好啊|你好在吗|哈喽你好|hello|hi|hey|Hi|HI|你好呀|嗨呀|hiya)', tn):
        return 'chat'

    return 'unknown'


def _generate_reply(intent, items, remarks, text, goods, cart_items):
    """根据意图 + 槽位，生成自然一点的回复（不再是固定话术）。"""
    tn = _normalize(text)
    cart_total = sum(float(x.get('price', 0) or 0) * int(x.get('count', 1) or 1) for x in (cart_items or []))
    cart_count = sum(int(x.get('count', 1) or 1) for x in (cart_items or []))
    # 把本次 items 也计进来（用于 order 意图的"合计"播报）
    extra_total = 0
    if intent == 'order':
        for it in items:
            g = next((x for x in goods if x.get('id') == it['id']), None)
            if g:
                extra_total += float(g.get('price', 0) or 0) * int(it.get('qty') or 1)
    total_after = cart_total + extra_total

    # —— 每种意图的"可理解"自然回复 ——
    if intent == 'cancel':
        if cart_items:
            return f'已为您清空购物车（共 {cart_count} 件商品，合计¥{cart_total:.2f}）。您可以重新开始点单。'
        return '购物车已经是空的啦。您可以直接告诉我想点什么，例如"来一杯拿铁"。'

    if intent == 'confirm':
        if not cart_items:
            return '购物车里还没有商品哦。您可以先告诉我想点的商品，再说"确认下单"。'
        names = '、'.join([f"{x.get('name','')}×{x.get('count',1)}" for x in cart_items])
        msg = f'好的，已确认您的订单：{names}，合计¥{cart_total:.2f}。请扫码完成支付。'
        return msg

    if intent == 'remove':
        if items:
            names = '、'.join([x['name'] for x in items])
            return f'好的，已为您从购物车中移除：{names}。'
        # 没说具体商品
        if cart_items:
            names = '、'.join([f"{x.get('name','')}×{x.get('count',1)}" for x in cart_items])
            return f'当前购物车里有：{names}。请问您想去掉哪一个？'
        return '购物车里现在没有商品，无法删除哦。您可以先告诉我想点什么。'

    if intent == 'price':
        if items:
            lines = []
            for it in items:
                g = next((x for x in goods if x.get('id') == it['id']), None)
                if g:
                    lines.append(f"{g.get('name','')} ¥{g.get('price',0)}")
            names_line = '；'.join(lines)
            hint = '需要直接帮您加入购物车吗？说"确认下单"就可以支付啦。'
            return f'{names_line}。{hint}'
        return '请问您想了解哪个商品的价格？可以直接说"拿铁多少钱"。'

    if intent == 'menu':
        # 按分类聚合（更自然）
        groups = {}
        for g in goods or []:
            key = g.get('cate') or '其他'
            groups.setdefault(key, []).append(g.get('name', ''))
        label_map = {
            'espresso': '浓缩咖啡类',
            'latte': '拿铁/奶咖类',
            'dessert': '甜品小食',
            'drink': '非咖饮品',
        }
        parts = []
        for key, names in groups.items():
            label = label_map.get(key, key)
            if names:
                parts.append(f"{label}有：{'、'.join(names[:6])}")
        if not parts:
            return '我们有咖啡、甜品和饮品，您可以直接说商品名，例如"来杯拿铁"。'
        body = '；'.join(parts)
        return f'好的，我们目前提供的品类如下——{body}。您想点哪一款？'

    if intent == 'recommend':
        # 挑出 3 个"招牌"候选（选价格中档的 + 第一个甜品）
        coffee = [g for g in goods if g.get('cate') in ('latte', 'espresso')]
        desserts = [g for g in goods if g.get('cate') == 'dessert']
        drinks = [g for g in goods if g.get('cate') == 'drink']
        picks = []
        if coffee:
            picks.append(coffee[len(coffee) // 2])  # 选中间那个（非最贵也非最便宜）
        if desserts:
            picks.append(desserts[0])
        if drinks:
            picks.append(drinks[0])
        if picks:
            names = '、'.join([f"{g.get('name','')}(¥{g.get('price',0)})" for g in picks])
            return f'推荐您尝试：{names}。您可以直接告诉我要哪个，例如"来一杯拿铁，两份提拉米苏"。'
        return '我们的招牌有拿铁咖啡、焦糖玛奇朵、提拉米苏、美式咖啡，您想尝试哪一款？'

    if intent == 'order':
        if not items:
            return '好的，不过我还没听清具体是哪个商品哦。可以再说一次商品名，例如"美式咖啡一杯"。'
        names = '、'.join([f"{x['name']}×{x['qty']}" for x in items])
        remark_part = ''
        if remarks:
            remark_part = f'，备注：{" ".join(remarks)}'
        after = f'本次加入后合计¥{total_after:.2f}' if total_after > 0 else '合计 ¥0'
        # 更自然的说法，随机化一下（不随机；用语气变化）
        variants = [
            f'好的，已为您下单：{names}{remark_part}。{after}。还需要其他商品吗？',
            f'收到，已经为您加入：{names}{remark_part}。{after}。可以继续点或说"确认下单"。',
            f'没问题～{names}已记录{remark_part}。{after}。还要点别的吗？',
        ]
        # 用文本长度做一个伪随机（保持稳定，不每次跳）
        idx = (len(text) + sum(ord(c) for c in text)) % len(variants)
        return variants[idx]

    if intent == 'remark':
        if remarks:
            hint = f'备注信息：{" ".join(remarks)}。'
            if cart_items:
                return f'好的，已记录{hint}我会附在您当前订单上。'
            return f'好的，{hint}不过您还没有选商品哦，可以直接告诉我您想点什么。'
        return '好的。'

    if intent == 'chat':
        return '你好！欢迎来到 Kora Zola。请问想点什么呢？可以直接说商品名，比如"一杯拿铁，少糖"。'

    # unknown：给出最像商品的提示，而不是说听不懂
    # 把所有商品名中 2 字以上的子词做一次是否存在的检查；如果有些关键词命中，则反问确认
    possible = []
    for g in goods or []:
        name_n = _normalize(g.get('name', ''))
        if not name_n:
            continue
        # 文本里是否包含商品名中的任意 >=2 字片段
        for i in range(len(name_n)):
            for j in range(i + 2, len(name_n) + 1):
                sub = name_n[i:j]
                if len(sub) >= 2 and sub in tn:
                    if g not in possible:
                        possible.append(g)
                    break
            if g in possible:
                break
    if possible:
        names = '、'.join([g.get('name', '') for g in possible[:3]])
        extra = '等' if len(possible) > 3 else ''
        return f'您是不是想点：{names}{extra}？请告诉我具体是哪一个，或者直接说商品全名+数量，例如"拿铁一杯"。'
    # 完全没商品信息 → 给出友好引导菜单
    categories = {}
    for g in goods or []:
        key = g.get('cate') or '其他'
        categories.setdefault(key, []).append(g.get('name', ''))
    examples = []
    for key, names in categories.items():
        if names:
            examples.append(names[0])
    return f'抱歉我没能准确理解。您可以直接告诉我：商品名 + 数量，例如"来一杯拿铁"、"两份提拉米苏"。目前有{", ".join(examples[:4])} 等可选。'


def local_parse(text, goods, cart_items):
    """轻量但可理解的本地规则引擎：
    1) 抽取槽位（商品+数量、备注）
    2) 意图判断（受文本 + 槽位共同影响）
    3) 生成自然回复（不再是硬编码的单一话术）
    """
    t = text.strip()
    if not t:
        return {'intent': 'unknown', 'items': [], 'remarks': [],
                'reply': '请告诉我您想点什么。', 'openCheckout': False}

    # 1) 商品识别（支持多商品）
    items = _smart_match_items(t, goods, cart_items or [])

    # 2) 备注识别
    remarks = _extract_remarks_smart(t)

    # 3) 是否只有备注（用于区分"只给备注"的意图）
    # 条件：有备注词 + 没有商品命中 + 文本中也没有其他强关键词
    has_remark_only = bool(remarks) and not bool(items)

    # 4) 意图判断
    intent = _classify_intent(t, items, has_remark_only, bool(cart_items))

    # 5) 生成自然回复
    reply = _generate_reply(intent, items, remarks, t, goods or [], cart_items or [])

    # 6) 如果是 confirm，标记 openCheckout=True（前端会据此打开支付弹窗）
    open_checkout = (intent == 'confirm' and bool(cart_items))

    # 7) 对于"推荐 / 菜单 / 价格"，也把命中的 items 带上，便于前端给用户点击确认下单
    # （price 已经在 items 里了；recommend/menu 则没有 items，这里不做强塞）

    return {
        'intent': intent,
        'items': items,
        'remarks': remarks,
        'reply': reply,
        'openCheckout': open_checkout,
    }


# ============================================================
# 主入口：/api/ai/chat
# ============================================================

def ai_chat(payload):
    """
    payload 期望：
      {
        "text": "用户说的一句话",
        "goods": [ {id, name, price, cate}... ],   // 当前门店商品列表
        "cart":  [ {id, name, price, count}... ] // 购物车现状
      }
    返回：{ "source": "llm"|"local", "engine": "...", ...意图JSON }
    """
    text = str(payload.get('text', '') or '').strip()
    goods = payload.get('goods') or []
    cart_items = payload.get('cart') or []

    if not text:
        return {'source': 'local', 'engine': 'fallback-empty',
                'intent': 'unknown', 'items': [], 'remarks': [],
                'reply': '请输入您想说的内容。', 'openCheckout': False}

    # 1) 先尝试大模型
    ai_cfg = CONFIG.get('ai', {})
    use_llm = bool(ai_cfg.get('enabled')) and bool(ai_cfg.get('apiKey')) and bool(ai_cfg.get('apiBase')) and bool(ai_cfg.get('model'))
    result = None
    source = 'local'
    engine = 'local-rules'

    if use_llm:
        goods_json = json.dumps(goods, ensure_ascii=False)
        cart_json = json.dumps(cart_items, ensure_ascii=False)
        sys_prompt = SYSTEM_PROMPT.replace('__GOODS_JSON_PLACEHOLDER__', goods_json) \
                                   .replace('__CART_JSON_PLACEHOLDER__', cart_json)
        try:
            t0 = time.time()
            raw = call_llm(sys_prompt, text)
            # 大模型有时会套 markdown ```json ... ```
            raw_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.IGNORECASE | re.DOTALL)
            parsed = json.loads(raw_clean)
            # 关键字段给默认值
            parsed.setdefault('items', [])
            parsed.setdefault('remarks', [])
            parsed.setdefault('reply', '好的。')
            parsed.setdefault('intent', 'unknown')
            parsed.setdefault('openCheckout', False)
            result = parsed
            source = 'llm'
            engine = CONFIG['ai'].get('model', 'unknown-model')
            log('OK', f'LLM 响应 OK ({time.time()-t0:.2f}s) intent={parsed.get("intent")}')
        except Exception as e:
            log('WARN', f'LLM 调用失败，回退到本地解析：{e}')

    # 2) 降级本地解析
    if result is None:
        result = local_parse(text, goods, cart_items)
        log('INFO', f'本地解析 intent={result.get("intent")} items={len(result.get("items",[]))}')

    # 归一化：保证 items 中每项都带 id/name/qty；如果大模型只给了 name，自动从 goods 反查 id
    if isinstance(result.get('items'), list):
        normalized_items = []
        for it in result['items']:
            name = str(it.get('name') or '').strip()
            gid = it.get('id')
            qty = int(it.get('qty') or 1)
            if qty <= 0:
                qty = 1
            if not gid and name:
                # 从 goods 里用名称反查
                hit = next((g for g in goods if str(g.get('name', '')) == name), None)
                if not hit:
                    # 模糊匹配
                    for g in goods:
                        gn = str(g.get('name', ''))
                        if name and (name in gn or gn in name):
                            hit = g
                            break
                if hit:
                    gid = hit['id']
                    name = hit['name']
            if gid is not None:
                normalized_items.append({'id': gid, 'name': name, 'qty': qty})
        result['items'] = normalized_items

    out = {
        'source': source,
        'engine': engine,
    }
    out.update(result)
    return out


# ============================================================
# HTTP 处理
# ============================================================

class Handler(BaseHTTPRequestHandler):
    # 静默日志（可改为覆盖 log_message 以自定义输出）
    def log_message(self, format, *args):
        log('INFO', '%s - %s' % (self.client_address[0], format % args))

    # ---- 基础工具 ----
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_file(self, path):
        if not os.path.isfile(path):
            self.send_response(404)
            self.end_headers()
            try:
                self.wfile.write(b'Not Found')
            except Exception:
                pass
            return
        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            # HTML/JS/CSS 兜底
            if path.endswith('.html'):
                mime = 'text/html; charset=utf-8'
            elif path.endswith('.js'):
                mime = 'application/javascript; charset=utf-8'
            elif path.endswith('.css'):
                mime = 'text/css; charset=utf-8'
            elif path.endswith('.json'):
                mime = 'application/json; charset=utf-8'
            else:
                mime = 'application/octet-stream'
        try:
            with open(path, 'rb') as f:
                body = f.read()
        except Exception as e:
            self._send_json({'ok': False, 'error': str(e)}, 500)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _read_body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode('utf-8') or '{}')
        except Exception as e:
            return {'__error': str(e)}

    # ---- 路由 ----
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ('/', ''):
            self._send_file(os.path.join(ROOT_DIR, 'index.html'))
            return
        if path == '/api/health':
            self._send_json({'ok': True, 'ts': int(time.time() * 1000),
                              'aiEnabled': CONFIG.get('ai', {}).get('enabled', False),
                              'hasKey': bool(CONFIG.get('ai', {}).get('apiKey', ''))})
            return
        if path == '/api/config':
            safe_cfg = json.loads(json.dumps(CONFIG))
            # 隐藏 key 的中间部分（前端仍可保存整个）
            try:
                key = safe_cfg['ai']['apiKey']
                if key and len(key) > 8:
                    safe_cfg['ai']['apiKey'] = key[:4] + '****' + key[-4:]
            except Exception:
                pass
            self._send_json({'ok': True, 'config': safe_cfg})
            return
        # 其他路径当作静态资源
        file_path = os.path.join(ROOT_DIR, unquote(path).lstrip('/'))
        # 防止目录穿越
        real = os.path.realpath(file_path)
        if not real.startswith(os.path.realpath(ROOT_DIR)):
            self._send_json({'ok': False, 'error': 'invalid path'}, 400)
            return
        self._send_file(file_path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()
        if isinstance(body, dict) and body.get('__error'):
            self._send_json({'ok': False, 'error': 'bad JSON: ' + body['__error']}, 400)
            return

        if path == '/api/ai/chat':
            try:
                result = ai_chat(body or {})
                self._send_json({'ok': True, **result})
            except Exception as e:
                log('ERR', 'ai_chat 异常: ' + str(e))
                self._send_json({'ok': False, 'error': str(e)}, 500)
            return

        if path == '/api/ai/test':
            # 用于在设置中测试大模型连通性
            try:
                raw = call_llm('你是一个测试助手，只输出 JSON。',
                                '请返回 {"ping":"pong"}')
                raw_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(),
                                   flags=re.IGNORECASE | re.DOTALL)
                data = json.loads(raw_clean)
                self._send_json({'ok': True, 'echo': data})
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, 500)
            return

        self._send_json({'ok': False, 'error': 'unknown route'}, 404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/config':
            body = self._read_body()
            if isinstance(body, dict) and body.get('__error'):
                self._send_json({'ok': False, 'error': 'bad JSON'}, 400)
                return
            if not isinstance(body, dict):
                self._send_json({'ok': False, 'error': 'body must be object'}, 400)
                return
            with _config_lock:
                merged = json.loads(json.dumps(DEFAULT_CONFIG))
                _deep_merge(merged, CONFIG)
                _deep_merge(merged, body)
                # 更新全局
                CONFIG.clear()
                CONFIG.update(merged)
                save_config(CONFIG)
            self._send_json({'ok': True, 'config': CONFIG})
            return
        self._send_json({'ok': False, 'error': 'unknown route'}, 404)


def main():
    parser = argparse.ArgumentParser(description='Kora Zola AI Server')
    parser.add_argument('--host', default=None, help='绑定地址（默认读取 config.json，再默认 0.0.0.0）')
    parser.add_argument('--port', type=int, default=None, help='端口（默认读取 config.json，再默认 8000）')
    args = parser.parse_args()

    host = args.host or CONFIG.get('server', {}).get('host') or '0.0.0.0'
    port = args.port or int(CONFIG.get('server', {}).get('port') or 8000)

    # 同步写回 config.json 中的 server 项（便于下次用）
    if args.host or args.port:
        CONFIG.setdefault('server', {})
        CONFIG['server']['host'] = host
        CONFIG['server']['port'] = port
        save_config(CONFIG)

    httpd = ThreadingHTTPServer((host, port), Handler)
    ai_cfg = CONFIG.get('ai', {})
    has_key = bool(ai_cfg.get('apiKey'))
    log('INFO', '=' * 60)
    log('INFO', 'Kora Zola AI 点单服务启动')
    log('INFO', f'前端页面:     http://{host}:{port}/  (或 http://127.0.0.1:{port}/)')
    log('INFO', f'顾客点单页:   http://{host}:{port}/customer.html')
    log('INFO', f'AI 接口:      POST http://127.0.0.1:{port}/api/ai/chat')
    log('INFO', f'配置文件:     {CONFIG_PATH}')
    log('INFO', f'大模型:       {"启用" if ai_cfg.get("enabled") else "禁用"} '
                f'| base={ai_cfg.get("apiBase")} | model={ai_cfg.get("model")}')
    if not has_key:
        log('WARN', '⚠ 尚未配置 API Key，将使用本地规则解析（也能点单，但不够智能）。')
        log('WARN', '  请在首页 → 设置 → AI助手中填写您的 Key 后保存。')
    log('INFO', '按 Ctrl+C 停止服务')
    log('INFO', '=' * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log('WARN', '服务已停止')
        httpd.server_close()
        sys.exit(0)


if __name__ == '__main__':
    main()
