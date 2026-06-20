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


def local_parse(text, goods, cart_items):
    """本地规则解析，返回与大模型一致的 JSON schema。"""
    t = text.strip()
    tn = _normalize(t)
    items = []
    remarks = _extract_remarks(t)

    # 意图判断优先级
    # 1. 取消 / 清空
    if re.search(r'(取消|不要了|算了|清空|退掉|不要这个|重来)', tn):
        return {'intent': 'cancel', 'items': [], 'remarks': remarks,
                'reply': '好的，已为您清空当前订单。', 'openCheckout': False}

    # 2. 确认 / 结账
    if re.search(r'(确认|结账|买单|支付|付款|就这|就这样|好了|够了|提交|下单|checkout|done|ok|是的|对|好的|来结账)', tn):
        if not cart_items:
            return {'intent': 'confirm', 'items': [], 'remarks': remarks,
                    'reply': '您还没有选择商品哦。', 'openCheckout': False}
        names = [x.get('name', '') for x in cart_items]
        total = round(sum(float(x.get('price', 0) or 0) * int(x.get('count', 1) or 1) for x in cart_items), 2)
        return {'intent': 'confirm', 'items': [], 'remarks': remarks,
                'reply': f'好的，已确认：{"、".join(names)}，合计¥{total:.2f}。请扫码完成支付。',
                'openCheckout': True}

    # 3. 删除 / 去掉
    if re.search(r'(去掉|不要|删除|移除|退|减去|少一份|少一个)', tn):
        items = _match_goods(t, goods)
        if items:
            names = [x['name'] for x in items]
            return {'intent': 'remove', 'items': items, 'remarks': remarks,
                    'reply': f'好的，已为您去掉：{"、".join(names)}。', 'openCheckout': False}
        # 没提到具体商品
        return {'intent': 'remove', 'items': [], 'remarks': remarks,
                'reply': '请问您想去掉哪一个呢？', 'openCheckout': False}

    # 4. 推荐
    if re.search(r'(推荐|招牌|热门|有什么|什么好|喝点什么|吃点什么|来点什么|suggest|recommend|popular)', tn):
        return {'intent': 'recommend', 'items': [], 'remarks': remarks,
                'reply': '我们的招牌有：拿铁咖啡、焦糖玛奇朵、提拉米苏、美式咖啡，您想尝试哪一款？', 'openCheckout': False}

    # 5. 价格询问
    price_m = re.search(r'(.*?)(多少钱|价格|价位|多少元|几元|多少钱一杯|how much|price)', tn)
    if price_m:
        items = _match_goods(price_m.group(1) or tn, goods)
        if items:
            p = items[0]
            # 查原商品价格
            detail = next((x for x in goods if x.get('id') == p['id']), None)
            price = detail.get('price', 0) if detail else 0
            return {'intent': 'price', 'items': items, 'remarks': remarks,
                    'reply': f"{p['name']}的价格是¥{price}。需要帮您下单吗？", 'openCheckout': False}

    # 6. 菜单查询
    if re.search(r'(有什么(咖啡|饮品|甜品|甜点|小食|蛋糕|点心)|菜单|menu|list)', tn):
        return {'intent': 'menu', 'items': [], 'remarks': remarks,
                'reply': '我们有咖啡类（拿铁、美式、意式浓缩、卡布奇诺、焦糖玛奇朵、生椰拿铁）、甜品（提拉米苏、蔓越莓司康）、饮品（鲜榨橙汁、气泡水）。', 'openCheckout': False}

    # 7. 加购商品
    items = _match_goods(t, goods)
    if items:
        names = [f"{x['name']}×{x['qty']}" for x in items]
        return {'intent': 'order', 'items': items, 'remarks': remarks,
                'reply': f"好的，已为您添加：{'、'.join(names)}。需要继续点单或直接确认下单吗？",
                'openCheckout': False}

    # 8. 纯备注
    if remarks:
        return {'intent': 'remark', 'items': [], 'remarks': remarks,
                'reply': f"好的，备注已记录：{' '.join(remarks)}。", 'openCheckout': False}

    # 9. 问候
    if len(t) <= 10 and re.search(r'(你好|您好|在吗|哈喽|嗨|hello|hi|hey)', tn):
        return {'intent': 'chat', 'items': [], 'remarks': [],
                'reply': '你好！欢迎来到 Kora Zola，请问想点什么呢？可以直接告诉我商品名，例如「一杯拿铁」。',
                'openCheckout': False}

    # 10. 无法理解
    return {'intent': 'unknown', 'items': [], 'remarks': [],
            'reply': '抱歉我没听清楚，请直接告诉我商品名，例如「一杯拿铁」或者「两份提拉米苏」。',
            'openCheckout': False}


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
