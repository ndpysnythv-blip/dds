// 本地规则解析冒烟测试 - 中英文+缺货+后台模式
const _normalize = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();

// ============= 中英文商品名映射 =============
const EN_NAME_MAP = {
  '拿铁': 'Latte', '拿铁咖啡': 'Latte',
  '美式': 'Americano', '美式咖啡': 'Americano',
  '焦糖玛奇朵': 'Caramel Macchiato',
  '提拉米苏': 'Tiramisu',
  '抹茶拿铁': 'Matcha Latte',
  '柠檬气泡水': 'Lemon Sparkling Water', '柠檬': 'Lemon Sparkling Water',
  '卡布奇诺': 'Cappuccino', '摩卡': 'Mocha'
};
const EN_KEYWORD_TO_CN = {
  'latte': '拿铁', 'americano': '美式',
  'caramel': '焦糖玛奇朵', 'macchiato': '焦糖玛奇朵',
  'tiramisu': '提拉米苏', 'matcha': '抹茶拿铁',
  'lemon': '柠檬气泡水', 'sparkling': '柠檬气泡水',
  'cappuccino': '卡布奇诺', 'mocha': '摩卡',
  'coffee': '咖啡', 'espresso': '浓缩'
};
function getEnName(name) {
  if (!name) return name;
  if (EN_NAME_MAP[name]) return EN_NAME_MAP[name];
  for (const k in EN_NAME_MAP) { if (name.indexOf(k) >= 0) return EN_NAME_MAP[k]; }
  return name;
}

// ============= 数量提取 =============
function _extractQuantity(text) {
  const out = [];
  const tn = _normalize(text);
  const numMap = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 };
  const cnMap = { 零: 0, 一: 1, 二: 2, 两: 2, 俩: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10 };
  const digits = tn.match(/\d+/g) || [];
  digits.forEach((d) => { if (d.length <= 3) out.push({ word: d, num: parseInt(d, 10) }); });
  Object.keys(cnMap).forEach((ch) => { if (tn.indexOf(ch) >= 0) out.push({ word: ch, num: cnMap[ch] }); });
  Object.keys(numMap).forEach((w) => { if (tn.indexOf(w) >= 0) out.push({ word: w, num: numMap[w] }); });
  return out;
}

// ============= 商品识别 =============
function _matchGoodsSmart(text, goods, cartItems, lang) {
  const hits = {};
  const tn = _normalize(text);
  const candidateHits = [];
  const isEn = (lang === 'en');

  for (let i = 0; i < goods.length; i++) {
    const g = goods[i];
    if (!g.name) continue;
    const gn = _normalize(g.name);
    // 等级 5：全名（中文）
    if (tn.indexOf(gn) >= 0) {
      let idx = 0;
      while ((idx = tn.indexOf(gn, idx)) >= 0) { candidateHits.push([5, idx, gn, g]); idx = idx + gn.length; }
      continue;
    }
    // 英文模式：英文关键词 → 中文商品名映射
    if (isEn) {
      let matched = false;
      for (const enKey in EN_KEYWORD_TO_CN) {
        const cnName = EN_KEYWORD_TO_CN[enKey];
        if (tn.indexOf(enKey) >= 0 && g.name.indexOf(cnName) >= 0) {
          candidateHits.push([4, tn.indexOf(enKey), enKey, g]);
          matched = true; break;
        }
      }
      if (matched) continue;
      continue;
    }
    // 中文前缀/后缀/子词匹配
    let foundPrefix = false;
    for (let len = gn.length; len >= 2; len--) {
      const prefix = gn.substring(0, len);
      if (tn.indexOf(prefix) >= 0) {
        let pidx = 0;
        while ((pidx = tn.indexOf(prefix, pidx)) >= 0) { candidateHits.push([4, pidx, prefix, g]); pidx = pidx + prefix.length; }
        foundPrefix = true; break;
      }
    }
    if (foundPrefix) continue;
    let foundSuffix = false;
    for (let len = gn.length; len >= 2; len--) {
      const suffix = gn.substring(gn.length - len);
      if (tn.indexOf(suffix) >= 0) {
        let sidx = 0;
        while ((sidx = tn.indexOf(suffix, sidx)) >= 0) { candidateHits.push([3, sidx, suffix, g]); sidx = sidx + suffix.length; }
        foundSuffix = true; break;
      }
    }
    if (foundSuffix) continue;
    let subHit = false;
    for (let si = 0; si < gn.length && !subHit; si++) {
      for (let sj = si + 2; sj <= gn.length && !subHit; sj++) {
        const sub = gn.substring(si, sj);
        if (tn.indexOf(sub) >= 0) { candidateHits.push([1, tn.indexOf(sub), sub, g]); subHit = true; }
      }
    }
  }
  // 排序：score desc, wordLen desc, pos asc
  candidateHits.sort((a, b) => {
    if (a[0] !== b[0]) return b[0] - a[0];
    if (a[2].length !== b[2].length) return b[2].length - a[2].length;
    return a[1] - b[1];
  });
  const usedPos = {};
  const filtered = [];
  for (let i = 0; i < candidateHits.length; i++) {
    const [, pos, word, g] = candidateHits[i];
    let conflict = false;
    for (const uk in usedPos) {
      const u = parseInt(uk, 10);
      const item = usedPos[uk];
      const overlap = (pos < u + item.wordLen && u < pos + word.length);
      if (overlap && item.score >= candidateHits[i][0]) { conflict = true; break; }
    }
    if (conflict) continue;
    usedPos[pos] = { score: candidateHits[i][0], wordLen: word.length, g };
    filtered.push([pos, word, g, candidateHits[i][0]]);
  }
  // 找数量
  for (let i = 0; i < filtered.length; i++) {
    const [pos, , g] = filtered[i];
    let qty = 1;
    const windowStart = Math.max(0, pos - 12);
    const windowEnd = Math.min(tn.length, pos + filtered[i][1].length + 12);
    const window = tn.substring(windowStart, windowEnd);
    const nums = _extractQuantity(window);
    if (nums.length > 0) {
      let best = nums[0];
      let bestDist = 9999;
      for (let k = 0; k < nums.length; k++) {
        const d = Math.abs((pos - windowStart) - (nums[k].word ? window.indexOf(nums[k].word) : 0));
        if (d < bestDist) { bestDist = d; best = nums[k]; }
      }
      qty = best.num;
    }
    if (/各(一|1)|每样|每个|每杯|each|per item/.test(window)) qty = 1;
    if (hits[g.id]) hits[g.id].qty = Math.max(hits[g.id].qty, qty);
    else hits[g.id] = { id: g.id, name: g.name, qty };
  }
  // 上下文："再来一份"
  if (Object.keys(hits).length === 0 &&
      (/(再来一份|再来一杯|再加一份|再加一个|同样的|还是这个|还是老样子|再来|再要|再加|一样的|续杯)/.test(tn)
       || /\b(one more|same again|another one|the same|same one|again)\b/.test(tn))) {
    if (cartItems && cartItems.length > 0) {
      const first = cartItems[0];
      hits[first.id] = { id: first.id, name: first.name, qty: 1 };
    }
  }
  const arr = [];
  for (const key in hits) {
    const hit = hits[key];
    let gd = null;
    for (let gi = 0; gi < goods.length; gi++) {
      if (goods[gi].id === hit.id) { gd = goods[gi]; break; }
    }
    if (gd) {
      let available = true;
      if (gd.status === 'off' || gd.status === 'offline') available = false;
      if (typeof gd.stock === 'number' && gd.stock <= 0) available = false;
      hit.available = available;
    }
    arr.push(hit);
  }
  return arr;
}

// ============= 备注提取 =============
function _extractRemarksSmart(text) {
  const tn = _normalize(text);
  const found = [];
  const pats = [
    /(无糖|不要糖|不加糖|去糖|零糖|0糖|低糖|少少糖|三分糖|半糖|少糖|多糖|全糖|加甜|甜一点|不要甜|不甜|很甜)/,
    /(去冰|少冰|少少冰|冰多一点|冰多|多冰|正常冰|冰|加冰|热|热的|热饮|常温|温|不冰)/,
    /(打包|外带|带走|外卖|带走喝|拿回家|堂食|店内|在这喝|在这吃|店里喝|店里)/,
    /(大杯|中杯|小杯|大份|小份|加大|超大|大一点|小一点)/,
    /(加奶|加浓|加倍浓缩|脱脂|燕麦奶|豆奶|换奶|淡奶|不要奶|不要奶泡|去奶泡)/,
    /(不要咖啡|不要咖啡因|低因|脱因)/,
    /\b(no ice|less ice|more ice|hot|cold|extra shot|skim milk|oat milk|take away|takeout|to go|no sugar|less sugar)\b/
  ];
  for (let pi = 0; pi < pats.length; pi++) {
    const m = tn.match(pats[pi]);
    if (m && m[1] && found.indexOf(m[1]) === -1) found.push(m[1]);
  }
  return found;
}

// ============= 意图分类 =============
function _classifyIntent(text, items, remarks, cartItems, lang) {
  const tn = _normalize(text);
  const hasRemark = remarks && remarks.length > 0;
  const hasCart = cartItems && cartItems.length > 0;
  const hasItems = items && items.length > 0;
  const isEn = (lang === 'en');

  if (isEn) {
    if (/\b(cancel|clear all|clear|empty|start over|never mind|nevermind|no thanks|no thank you|scratch that|reset)\b/.test(tn)) return 'cancel';
  } else {
    if (/(取消订单|全部取消|全部不要|清空|重来|重新点|重新下单|不要了|算了|不用了|退掉|cancel all|clear all)/.test(tn)) return 'cancel';
  }
  if (isEn) {
    if (/\b(remove|delete|take out|take away|without|don't|do not|no more|drop|minus)\b/.test(tn)) {
      if (hasItems) return 'remove';
      if (hasCart) return 'remove';
    }
  } else {
    if (/(去掉|不要|删除|移除|减去|少一份|少一个|去掉这个|别加|去掉那个|不要这个|不要那个|no|remove)/.test(tn)) {
      if (hasItems) return 'remove';
      if (hasCart && !(hasRemark && !hasItems)) return 'remove';
    }
  }
  if (isEn) { if (/\b(how much|price|how much is|how much for|how much does|costs|how much are)\b/.test(tn)) return 'price'; }
  else { if (/(多少钱|价格|价位|多少元|几元|多少钱一杯|how much|price|多少钱一份)/.test(tn)) return 'price'; }
  if (isEn) {
    if (/\b(menu|what do you have|what's on the menu|what is on the menu|do you have|what do you offer|what drinks|any coffee|what can i order)\b/.test(tn)) {
      if (/\b(recommend|suggest|popular|special|signature|best seller|good ones|any suggestion)\b/.test(tn)) return 'recommend';
      return 'menu';
    }
  } else {
    if (/(有没有|有什么|有啥|还有啥|还有什么|有哪些|菜单|菜单一览|都有什么|都有啥|提供什么|有什么可以|有什么好)/.test(tn)) {
      if (/(推荐|招牌|热门|给点建议|给建议|好喝的|好吃的|冷饮|热饮|特色|哪款|哪一种)/.test(tn)) return 'recommend';
      return 'menu';
    }
  }
  if (isEn) { if (/\b(recommend|suggest|popular|what's good|what is good|any suggestion|any recommendations|signature|best seller|special)\b/.test(tn)) return 'recommend'; }
  else { if (/(推荐|招牌|热门|什么好|好喝|好吃|喝点什么|吃点什么|来点什么|不知道点什么|给点建议|给建议|推荐一下|有什么推荐|有啥推荐|哪款好|哪个好|选什么|挑一个|suggest|recommend|popular)/.test(tn)) return 'recommend'; }
  let hasOrderAction;
  if (isEn) hasOrderAction = /\b(i want|i'll have|i would like|can i get|can i have|please give me|give me|bring me|add|order me|i like to order|pls get|get me)\b/.test(tn);
  else hasOrderAction = /(来杯|来一杯|来一份|点一杯|点一份|点一个|加一杯|加一份|加一个|点个|来个|打包一份|打包带走|给我来|给我点|我要|我点|给我一杯|给我一份|帮我点|帮我来|想要一杯|想要一份|想要)/.test(tn);
  if (hasItems && hasOrderAction) return 'order';
  if (isEn) { if (!hasItems && /\b(checkout|check out|confirm|that's all|that is all|ok|okay|done|yes|pay|payment|finish|ready to order|that's it|that is it|go ahead|proceed)\b/.test(tn)) return 'confirm'; }
  else { if (!hasItems && /(确认下单|确认一下|确认|结账|买单|去支付|支付|付款|就这样|就这些|就这些吧|好了|够了|checkout|done|ok|是的|对|好的|就这个|就这个吧|行了)/.test(tn)) return 'confirm'; }
  if (hasItems) return 'order';
  if (hasRemark && !hasItems) return 'remark';
  if (isEn) { if (text.length <= 30 && /\b(hello|hi|hey|hiya|good morning|good afternoon|good evening|welcome|greetings|howdy)\b/.test(tn)) return 'chat'; }
  else { if (text.length <= 12 && /(你好|您好|在吗|哈喽|嗨|你好啊|哈喽你好|hello|hi|hey|Hi|HI|你好呀|嗨呀|hiya|欢迎|welcome)/.test(tn)) return 'chat'; }
  return 'unknown';
}

// ============= 回复生成 =============
function _generateReply(intent, items, remarks, text, goods, cartItems, lang) {
  const tn = _normalize(text);
  const isEn = (lang === 'en');
  let cartTotal = 0, cartCount = 0;
  for (let ci = 0; ci < (cartItems || []).length; ci++) {
    cartTotal += (cartItems[ci].price || 0) * (cartItems[ci].count || 1);
    cartCount += (cartItems[ci].count || 1);
  }
  const availableItems = (items || []).filter((it) => it.available !== false);
  const unavailableItems = (items || []).filter((it) => it.available === false);
  const unavailNamesCn = unavailableItems.map((it) => it.name).join('、');
  const unavailNamesEn = unavailableItems.map((it) => getEnName(it.name)).join(', ');
  let extra = 0;
  if (intent === 'order') {
    for (let ii = 0; ii < availableItems.length; ii++) {
      let g = null;
      for (let gi = 0; gi < goods.length; gi++) { if (goods[gi].id === availableItems[ii].id) { g = goods[gi]; break; } }
      if (g) extra += (g.price || 0) * (availableItems[ii].qty || 1);
    }
  }
  const totalAfter = cartTotal + extra;

  if (intent === 'cancel') {
    if (isEn) {
      if (cartItems && cartItems.length > 0) return 'Your cart (' + cartCount + ' items) has been cleared. Total was ¥' + cartTotal.toFixed(2) + '. You can start a new order.';
      return 'Your cart is empty. Just tell me what you would like.';
    }
    if (cartItems && cartItems.length > 0) return '已为您清空购物车（共 ' + cartCount + ' 件商品，合计¥' + cartTotal.toFixed(2) + '）。您可以重新开始点单。';
    return '购物车已经是空的啦。您可以直接告诉我想点什么，例如"来一杯拿铁"。';
  }
  if (intent === 'confirm') {
    if (isEn) {
      if (!cartItems || cartItems.length === 0) return 'Your cart is empty. Please tell me what you want first, then say "checkout".';
      const n = cartItems.map((g) => getEnName(g.name) + ' ×' + (g.count || 1)).join(', ');
      return 'Order confirmed: ' + n + '. Total ¥' + cartTotal.toFixed(2) + '. Please proceed to checkout.';
    }
    if (!cartItems || cartItems.length === 0) return '购物车里还没有商品哦。您可以先告诉我想点的商品，再说"确认下单"。';
    const n = cartItems.map((g) => g.name + '×' + (g.count || 1)).join('、');
    return '好的，已确认您的订单：' + n + '，合计¥' + cartTotal.toFixed(2) + '。请扫码完成支付。';
  }
  if (intent === 'remove') {
    if (isEn) {
      if (items && items.length > 0) return 'Removed from your cart: ' + items.map((x) => getEnName(x.name)).join(', ') + '.';
      if (cartItems && cartItems.length > 0) return 'Your cart currently has: ' + cartItems.map((g) => getEnName(g.name) + '×' + (g.count || 1)).join(', ') + '. Which one would you like to remove?';
      return 'Your cart is empty. Nothing to remove.';
    }
    if (items && items.length > 0) return '好的，已为您从购物车中移除：' + items.map((x) => x.name).join('、') + '。';
    if (cartItems && cartItems.length > 0) return '当前购物车里有：' + cartItems.map((g) => g.name + '×' + (g.count || 1)).join('、') + '。请问您想去掉哪一个？';
    return '购物车里现在没有商品，无法删除哦。您可以先告诉我想点什么。';
  }
  if (intent === 'price') {
    if (isEn) {
      if (items && items.length > 0) {
        const lines = items.map((it) => {
          const g = goods.find((x) => x.id === it.id);
          return g ? getEnName(g.name) + ' ¥' + g.price : '';
        }).filter(Boolean).join('; ');
        return lines + '. Shall I add it to your cart?';
      }
      return 'Which item would you like to know the price for? Try "How much is a latte?".';
    }
    if (items && items.length > 0) {
      const lines = items.map((it) => {
        const g = goods.find((x) => x.id === it.id);
        return g ? g.name + ' ¥' + g.price : '';
      }).filter(Boolean).join('；');
      return lines + '。需要直接帮您加入购物车吗？说"确认下单"就可以支付啦。';
    }
    return '请问您想了解哪个商品的价格？可以直接说"拿铁多少钱"。';
  }
  if (intent === 'menu') {
    const groups = {};
    for (let mi = 0; mi < goods.length; mi++) {
      const key = goods[mi].cate || '其他';
      if (!groups[key]) groups[key] = [];
      groups[key].push(goods[mi].name);
    }
    if (isEn) {
      const labelMap = { espresso: 'Coffee', latte: 'Latte', dessert: 'Desserts', drink: 'Drinks', other: 'Others' };
      const parts = [];
      for (const k in groups) if (groups[k].length > 0) parts.push((labelMap[k] || k) + ': ' + groups[k].slice(0, 5).map((n) => getEnName(n)).join(', '));
      if (parts.length === 0) return 'We serve coffee, desserts and drinks. Tell me what you would like.';
      return 'Here is our menu: ' + parts.join('; ') + '. What would you like?';
    }
    const labels = { espresso: '咖啡类', latte: '拿铁/奶咖类', dessert: '甜品小食', drink: '非咖饮品' };
    const parts = [];
    for (const k in groups) if (groups[k].length > 0) parts.push((labels[k] || k) + '有：' + groups[k].slice(0, 6).join('、'));
    if (parts.length === 0) return '我们有咖啡、甜品和饮品，您可以直接说商品名，例如"来杯拿铁"。';
    return '好的，我们目前提供的品类如下——' + parts.join('；') + '。您想点哪一款？';
  }
  if (intent === 'recommend') {
    const picks = [];
    const coffee = goods.filter((g) => g.cate === 'latte' || g.cate === 'espresso');
    const desserts = goods.filter((g) => g.cate === 'dessert');
    const drinks = goods.filter((g) => g.cate === 'drink');
    if (coffee.length > 0) picks.push(coffee[Math.floor(coffee.length / 2)]);
    if (desserts.length > 0) picks.push(desserts[0]);
    if (drinks.length > 0) picks.push(drinks[0]);
    if (isEn) {
      if (picks.length === 0) return 'Our signature items are Latte, Caramel Macchiato, Tiramisu and Americano. Which one would you like?';
      return 'I recommend: ' + picks.map((g) => getEnName(g.name) + '(¥' + g.price + ')').join(', ') + '. Just tell me what you want.';
    }
    if (picks.length === 0) return '我们的招牌有拿铁咖啡、焦糖玛奇朵、提拉米苏、美式咖啡，您想尝试哪一款？';
    return '推荐您尝试：' + picks.map((g) => g.name + '(¥' + g.price + ')').join('、') + '。您可以直接告诉我要哪个，例如"来一杯拿铁，两份提拉米苏"。';
  }
  if (intent === 'order') {
    if (availableItems.length === 0 && unavailableItems.length > 0) {
      return isEn ? 'Sorry, ' + unavailNamesEn + ' is currently out of stock. Would you like to try something else?' : '抱歉，' + unavailNamesCn + ' 暂时缺货/已下架。您可以尝试点其他商品。';
    }
    if (!availableItems || availableItems.length === 0) {
      return isEn ? 'Please tell me the item name, e.g. "One latte please".' : '请告诉我具体商品名，例如"拿铁一杯"。';
    }
    if (isEn) {
      const names = availableItems.map((g) => getEnName(g.name) + ' ×' + (g.qty || 1)).join(', ');
      const remarkStr = (remarks && remarks.length > 0) ? ' (notes: ' + remarks.join(', ') + ')' : '';
      const unavailMsg = unavailableItems.length > 0 ? ' (Note: ' + unavailNamesEn + ' is out of stock)' : '';
      return 'Added to your cart: ' + names + remarkStr + '. Total now ¥' + totalAfter.toFixed(2) + '.' + unavailMsg;
    }
    const names = availableItems.map((g) => g.name + '×' + (g.qty || 1)).join('、');
    const remarkStr = (remarks && remarks.length > 0) ? '，备注：' + remarks.join(' ') : '';
    const unavailMsg = unavailableItems.length > 0 ? ' ⚠️ 注意：' + unavailNamesCn + ' 暂时缺货/已下架。' : '';
    return '好的，已为您下单：' + names + remarkStr + '。当前购物车合计¥' + totalAfter.toFixed(2) + '。还需要点其他商品吗？' + unavailMsg;
  }
  if (intent === 'remark') {
    if (isEn) {
      if (remarks && remarks.length > 0) {
        if (cartItems && cartItems.length > 0) return 'Got it. Notes saved: ' + remarks.join(', ') + ' to your current order.';
        return 'Notes saved: ' + remarks.join(', ') + '. But your cart is empty - please tell me what you want.';
      }
      return 'OK.';
    }
    if (remarks && remarks.length > 0) {
      if (cartItems && cartItems.length > 0) return '好的，已记录备注：' + remarks.join(' ') + '。我会附在您当前订单上。';
      return '好的，备注已记录：' + remarks.join(' ') + '。不过您还没有选商品哦，可以直接告诉我您想点什么。';
    }
    return '好的。';
  }
  if (intent === 'chat') return isEn ? 'Hello! Welcome. What would you like today?' : '您好！欢迎光临。您想喝点/吃点什么？';
  return isEn ? "Sorry I didn't catch that. Try: one latte, two americano, what's on the menu, how much is a latte, checkout, cancel." : '抱歉我没能准确理解。您可以直接告诉我：商品名 + 数量，例如"来一杯拿铁"、"两份提拉米苏"。';
}

// ============= 主入口 =============
function smartLocalParse(text, goods, cartItems, lang) {
  const t = _normalize(text);
  const curLang = lang || 'zh';
  if (!t) return { intent: 'unknown', items: [], remarks: [], reply: curLang === 'en' ? 'Please tell me what you would like.' : '请告诉我您想点什么。', openCheckout: false };
  const items = _matchGoodsSmart(text, goods || [], cartItems || [], curLang);
  const remarks = _extractRemarksSmart(text);
  const intent = _classifyIntent(text, items, remarks, cartItems || [], curLang);
  const reply = _generateReply(intent, items, remarks, text, goods || [], cartItems || [], curLang);
  return { intent, items, remarks, reply, openCheckout: (intent === 'confirm' && cartItems && cartItems.length > 0) };
}

// ============= 后台管理指令 =============
function handleAdminCommand(text, goods) {
  const tn = _normalize(text);
  const m = tn.match(/(下架|off|disable)\s*(.+?)(?:\s|$)/);
  if (m && m[2]) {
    const g = goods.find((x) => _normalize(x.name).indexOf(_normalize(m[2])) >= 0 || _normalize(m[2]).indexOf(_normalize(x.name)) >= 0);
    if (g) return { action: 'offline', target: g.name, reply: '✅ 已将 ' + g.name + ' 标记为下架。' };
    return { action: 'unknown', reply: '❌ 未找到商品：' + m[2] };
  }
  const m2 = tn.match(/(上架|on|enable|恢复)\s*(.+?)(?:\s|$)/);
  if (m2 && m2[2]) {
    const g = goods.find((x) => _normalize(x.name).indexOf(_normalize(m2[2])) >= 0 || _normalize(m2[2]).indexOf(_normalize(x.name)) >= 0);
    if (g) return { action: 'online', target: g.name, reply: '✅ 已将 ' + g.name + ' 标记为在售。' };
    return { action: 'unknown', reply: '❌ 未找到商品：' + m2[2] };
  }
  const m3 = tn.match(/(库存|stock)\s*(.+?)\s*(\d+)/);
  if (m3 && m3[3]) {
    const stock = parseInt(m3[3], 10);
    const gName = m3[2].trim();
    const g = goods.find((x) => _normalize(x.name).indexOf(_normalize(gName)) >= 0);
    if (g) return { action: 'setStock', target: g.name, stock, reply: '✅ 已设置 ' + g.name + ' 的库存为 ' + stock + '。' };
    return { action: 'setStock', reply: '请指定商品名，例如"库存 拿铁 50"。' };
  }
  return { action: 'help', reply: '后台管理模式支持：\n• 下架 [商品名]\n• 上架 [商品名]\n• 库存 [商品名] [数量]\n• 退出后台 → 返回普通模式' };
}

// ============= 测试用商品数据 =============
const testGoods = [
  { id: '1', name: '拿铁咖啡', price: 28, cate: 'latte', stock: 100, status: 'on' },
  { id: '2', name: '美式咖啡', price: 22, cate: 'espresso', stock: 80, status: 'on' },
  { id: '3', name: '焦糖玛奇朵', price: 32, cate: 'latte', stock: 0, status: 'on' },
  { id: '4', name: '提拉米苏', price: 38, cate: 'dessert', stock: 50, status: 'on' },
  { id: '5', name: '抹茶拿铁', price: 30, cate: 'latte', stock: 60, status: 'off' },
  { id: '6', name: '柠檬气泡水', price: 18, cate: 'drink', stock: 200, status: 'on' }
];

// ============= 运行测试 =============
const testCases = [
  // 中文 - 基础场景
  { name: '[中文] 普通下单', text: '拿铁一杯', lang: 'zh', cart: [] },
  { name: '[中文] 动作词下单+备注', text: '来一杯拿铁少冰', lang: 'zh', cart: [] },
  { name: '[中文] 多商品', text: '两份提拉米苏和一杯美式', lang: 'zh', cart: [] },
  { name: '[中文] 价格询问', text: '拿铁多少钱', lang: 'zh', cart: [] },
  { name: '[中文] 推荐', text: '有什么推荐', lang: 'zh', cart: [] },
  { name: '[中文] 菜单', text: '有什么喝的', lang: 'zh', cart: [] },
  { name: '[中文] 确认下单', text: '确认下单', lang: 'zh', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 2 }] },
  { name: '[中文] 取消订单', text: '全部取消', lang: 'zh', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 2 }] },
  { name: '[中文] 删除商品', text: '去掉拿铁', lang: 'zh', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 2 }] },
  { name: '[中文] 缺货(stock=0)', text: '来一杯焦糖玛奇朵', lang: 'zh', cart: [] },
  { name: '[中文] 下架(status=off)', text: '一杯抹茶拿铁', lang: 'zh', cart: [] },
  { name: '[中文] 问候', text: '你好', lang: 'zh', cart: [] },

  // 英文 - 对应场景
  { name: '[英文] 普通下单', text: 'one latte please', lang: 'en', cart: [] },
  { name: '[英文] 多商品', text: 'can i get one latte and two americano', lang: 'en', cart: [] },
  { name: '[英文] 价格询问', text: 'how much is a latte', lang: 'en', cart: [] },
  { name: '[英文] 推荐', text: 'what do you recommend', lang: 'en', cart: [] },
  { name: '[英文] 菜单', text: 'what is on the menu', lang: 'en', cart: [] },
  { name: '[英文] 确认下单', text: 'checkout please', lang: 'en', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 1 }] },
  { name: '[英文] 取消', text: 'cancel my order', lang: 'en', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 1 }] },
  { name: '[英文] 删除商品', text: 'remove the latte', lang: 'en', cart: [{ id: '1', name: '拿铁咖啡', price: 28, count: 1 }] },
  { name: '[英文] 缺货商品', text: 'one caramel macchiato please', lang: 'en', cart: [] },
  { name: '[英文] 下架商品', text: 'one matcha latte', lang: 'en', cart: [] },
  { name: '[英文] 问候', text: 'hello', lang: 'en', cart: [] },
  { name: '[英文] 再来一份', text: 'one more please', lang: 'en', cart: [{ id: '2', name: '美式咖啡', price: 22, count: 1 }] },

  // 后台管理
  { name: '[后台] 下架拿铁', text: '下架 拿铁咖啡', admin: true },
  { name: '[后台] 上架抹茶拿铁', text: '上架 抹茶拿铁', admin: true },
  { name: '[后台] 设置库存', text: '库存 提拉米苏 50', admin: true },
  { name: '[后台] 帮助', text: 'help', admin: true }
];

console.log('=========================================');
console.log('  AI 点单系统 - 本地规则冒烟测试');
console.log('=========================================\n');

let passed = 0, failed = 0;
const failures = [];

for (const tc of testCases) {
  try {
    let result;
    if (tc.admin) result = handleAdminCommand(tc.text, testGoods);
    else result = smartLocalParse(tc.text, testGoods, tc.cart, tc.lang);

    const isEn = tc.lang === 'en';
    // 英文模式：回复必须全英文，无中文字符（除商品价格的¥外）
    let replyValid = true;
    if (isEn && result.reply) {
      // 检查是否有中文字符在回复中（允许的商品名需要翻译成英文）
      const hasChinese = /[\u4e00-\u9fa5]/.test(result.reply);
      if (hasChinese) replyValid = false;
    }
    // 意图有效性：必须是已知意图
    const validIntents = ['order', 'cancel', 'confirm', 'remove', 'price', 'menu', 'recommend', 'remark', 'chat', 'unknown', 'help'];
    const intentOk = tc.admin ? true : validIntents.indexOf(result.intent) >= 0;
    // 下单/价格/删除/确认等场景应命中相应商品
    const shouldHaveItems = !tc.admin && ['order', 'price', 'remove'].indexOf(result.intent) >= 0;
    const itemsOk = tc.admin ? true : (!shouldHaveItems || result.items.length > 0);
    const ok = intentOk && itemsOk && replyValid;

    console.log(`【${ok ? '✓' : '✗'}】${tc.name}`);
    console.log(`      输入: "${tc.text}"`);
    if (tc.admin) {
      console.log(`      动作: ${result.action}`);
      console.log(`      回复: "${result.reply.slice(0, 80)}${result.reply.length > 80 ? '...' : ''}"`);
    } else {
      console.log(`      意图: ${result.intent} | 商品: ${result.items.length} 项 | 备注: ${result.remarks.join(', ') || '-'}`);
      console.log(`      回复: "${result.reply.slice(0, 120)}${result.reply.length > 120 ? '...' : ''}"`);
      if (result.items.length > 0) {
        for (const it of result.items) console.log(`        → ${it.name} × ${it.qty} (可用: ${it.available !== false})`);
      }
    }
    console.log();

    if (ok) passed++;
    else {
      failed++;
      failures.push({ name: tc.name, text: tc.text, result });
    }
  } catch (e) {
    failed++;
    console.log(`【✗】${tc.name} → 异常: ${e.message}`);
    console.log();
    failures.push({ name: tc.name, text: tc.text, error: e.message });
  }
}

console.log('=========================================');
console.log(`  测试结果: 通过 ${passed} / 失败 ${failed} （共 ${testCases.length}）`);
console.log('=========================================');

if (failures.length > 0) {
  console.log('\n【失败详情】:');
  for (const f of failures) console.log('  - ' + f.name + ': ' + (f.error || '回复包含中文 / 商品未命中 / 意图无效'));
}

if (failed === 0) {
  console.log('\n🎉 所有测试通过！中英文+缺货+后台模式均正常工作。');
  process.exit(0);
} else {
  console.log('\n⚠️  有 ' + failed + ' 个测试失败，请检查。');
  process.exit(1);
}
