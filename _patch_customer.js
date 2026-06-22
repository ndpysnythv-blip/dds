// 替换 customer.html 中的 script 块为优化版
const fs = require('fs');
const path = '/workspace/customer.html';
const html = fs.readFileSync(path, 'utf-8');

const newScript = `  <script>
    // ============ 工具函数（稳定、低开销）============
    function safeGetLS(key, fb) {
      try { var r = localStorage.getItem(key); return r ? JSON.parse(r) : fb; }
      catch (e) { return fb; }
    }
    function safeSetLS(key, val) {
      try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {}
    }
    function debounce(fn, wait) {
      var t = null;
      return function () {
        var ctx = this, args = arguments;
        clearTimeout(t);
        t = setTimeout(function () { fn.apply(ctx, args); }, wait);
      };
    }
    var $cache = {};
    function $(id) { if (!$cache[id]) $cache[id] = document.getElementById(id); return $cache[id]; }

    // ============ 数据层 ============
    var DEFAULT_GOODS = [
      { id: 1, name: '\u7ecf\u5178\u610f\u5f0f\u6d53\u7f29', price: 18, stock: 99, cate: 'espresso', img: 'https://picsum.photos/id/431/200/150' },
      { id: 2, name: '\u7f8e\u5f0f\u5496\u5561', price: 22, stock: 99, cate: 'espresso', img: 'https://picsum.photos/id/766/200/150' },
      { id: 3, name: '\u62ff\u94c1\u5496\u5561', price: 28, stock: 99, cate: 'latte', img: 'https://picsum.photos/id/225/200/150' },
      { id: 4, name: '\u5361\u5e03\u5947\u8bfa', price: 30, stock: 99, cate: 'latte', img: 'https://picsum.photos/id/312/200/150' },
      { id: 5, name: '\u751f\u6930\u62ff\u94c1', price: 32, stock: 99, cate: 'latte', img: 'https://picsum.photos/id/433/200/150' },
      { id: 6, name: '\u63d0\u62c9\u7c73\u82cf', price: 26, stock: 50, cate: 'dessert', img: 'https://picsum.photos/id/292/200/150' },
      { id: 7, name: '\u8513\u8d8a\u8393\u53f8\u5eb7', price: 16, stock: 30, cate: 'dessert', img: 'https://picsum.photos/id/434/200/150' },
      { id: 8, name: '\u9c9c\u69a8\u6a59\u6c41', price: 20, stock: 20, cate: 'drink', img: 'https://picsum.photos/id/447/200/150' },
      { id: 9, name: '\u6c14\u6ce1\u6c34', price: 18, stock: 40, cate: 'drink', img: 'https://picsum.photos/id/488/200/150' },
      { id: 10, name: '\u7126\u7cd6\u739b\u5947\u6735', price: 35, stock: 99, cate: 'latte', img: 'https://picsum.photos/id/614/200/150' }
    ];
    var GOODS_DATA = safeGetLS('starCoffeeGoods', DEFAULT_GOODS);
    var GOODS_CACHE = {};
    (function () { for (var i = 0; i < GOODS_DATA.length; i++) GOODS_CACHE[GOODS_DATA[i].id] = GOODS_DATA[i]; })();
    var cart = {};
    var currentCate = 'all';
    var currentOrderId = null;
    var orderPollTimer = null;
    var _lastPickupCode = null;

    // ============ 渲染商品列表（DocumentFragment，单次回流）============
    function renderGoodsList() {
      var container = $('goods-list');
      if (!container) return;
      var list = currentCate === 'all' ? GOODS_DATA : GOODS_DATA.filter(function (it) { return it.cate === currentCate; });
      var items = list.filter(function (it) { return it.stock > 0; });
      var frag = document.createDocumentFragment();
      for (var i = 0; i < items.length; i++) {
        var g = items[i];
        var wrap = document.createElement('div');
        wrap.className = 'goods-card';
        wrap.innerHTML = '<img src="' + g.img + '" alt="' + g.name + '" class="w-full h-32 object-cover" loading="lazy">' +
          '<div class="p-3"><h3 class="font-medium text-coffee-dark text-sm mb-1">' + g.name + '</h3>' +
          '<div class="flex justify-between items-center"><span class="text-coffee-main font-bold">\u00a5' + (typeof g.price === 'number' ? g.price.toFixed(2) : '0.00') + '</span>' +
          '<div class="flex items-center gap-2">' +
          '<button class="add-minus-btn w-7 h-7 flex items-center justify-center rounded-full bg-coffee-cream text-coffee-dark btn-hover" data-id="' + g.id + '" data-action="minus"><i class="fa-solid fa-minus text-xs"></i></button>' +
          '<span class="text-sm w-6 text-center" id="count-' + g.id + '">' + (cart[g.id] ? cart[g.id].count : 0) + '</span>' +
          '<button class="add-minus-btn w-7 h-7 flex items-center justify-center rounded-full bg-coffee-main text-white btn-hover" data-id="' + g.id + '" data-action="plus"><i class="fa-solid fa-plus text-xs"></i></button>' +
          '</div></div></div>';
        frag.appendChild(wrap);
      }
      container.textContent = '';
      container.appendChild(frag);
    }

    // ============ 增量更新购物车显示（不重建列表）============
    function updateCartDisplay() {
      var cartList = []; var totalCount = 0, totalPrice = 0;
      var keys = Object.keys(cart);
      for (var i = 0; i < keys.length; i++) {
        var it = cart[keys[i]];
        if (it && it.count > 0) { cartList.push(it); totalCount += it.count; totalPrice += it.price * it.count; }
      }
      var countEl = $('cart-count'), totalEl = $('cart-total'), detailEl = $('cart-detail'), submitBtn = $('submit-order');
      if (countEl) countEl.textContent = totalCount;
      if (totalEl) totalEl.textContent = '\u00a5' + totalPrice.toFixed(2);
      if (detailEl) detailEl.textContent = cartList.length ? cartList.map(function (it) { return it.name + '\u00d7' + it.count; }).join('\uff0c') : '\u8d2d\u7269\u8f66\u662f\u7a7a\u7684';
      if (submitBtn) submitBtn.disabled = totalCount === 0;
      for (var j = 0; j < cartList.length; j++) {
        var el = document.getElementById('count-' + cartList[j].id);
        if (el) {
          el.textContent = cartList[j].count;
          el.classList.remove('num-pulse');
          void el.offsetWidth;
          el.classList.add('num-pulse');
        }
      }
      // 处理减到 0 的商品（不在 cartList 中，但需要归零显示）
      for (var k = 0; k < keys.length; k++) {
        if (cart[keys[k]] && cart[keys[k]].count === 0) {
          var el2 = document.getElementById('count-' + keys[k]);
          if (el2) el2.textContent = '0';
        }
      }
    }

    // ============ 事件委托：商品 +/- ============
    if ($('goods-list')) {
      $('goods-list').addEventListener('click', function (e) {
        var btn = e.target && e.target.closest ? e.target.closest('.add-minus-btn') : null;
        if (!btn) return;
        var goodsId = parseInt(btn.dataset.id, 10);
        var action = btn.dataset.action;
        var g = GOODS_CACHE[goodsId];
        if (!g) return;
        if (action === 'plus') { if (!cart[goodsId]) cart[goodsId] = { id: g.id, name: g.name, price: g.price, count: 0 }; cart[goodsId].count++; }
        else if (action === 'minus') { if (cart[goodsId] && cart[goodsId].count > 0) cart[goodsId].count--; }
        updateCartDisplay();
      });
    }

    // ============ 分类切换 ============
    document.querySelectorAll('.cate-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        currentCate = btn.dataset.cate;
        document.querySelectorAll('.cate-btn').forEach(function (b) {
          b.classList.remove('bg-coffee-main', 'text-white'); b.classList.add('bg-white', 'text-coffee-dark');
        });
        btn.classList.add('bg-coffee-main', 'text-white'); btn.classList.remove('bg-white', 'text-coffee-dark');
        renderGoodsList();
      });
    });

    // ============ 提交订单 ============
    if ($('submit-order')) {
      $('submit-order').addEventListener('click', function () {
        var cartList = Object.values(cart).filter(function (it) { return it.count > 0; });
        if (cartList.length === 0) return;
        var remark = $('order-remark') ? $('order-remark').value || '' : '';
        var totalPrice = 0; for (var i = 0; i < cartList.length; i++) totalPrice += cartList[i].price * cartList[i].count;
        var verifyCode = String(Math.floor(1000 + Math.random() * 9000));
        var newOrder = { id: Date.now(), goods: JSON.parse(JSON.stringify(cart)), total: totalPrice, remark: remark, createTime: Date.now(), verifyCode: verifyCode };
        currentOrderId = newOrder.id;
        var pending = safeGetLS('customerPendingOrders', []); pending.push(newOrder);
        safeSetLS('customerPendingOrders', pending);
        showWaitingPage(newOrder);
        cart = {};
        if ($('order-remark')) $('order-remark').value = '';
        updateCartDisplay();
      });
    }

    // ============ 等待支付页渲染 ============
    function showWaitingPage(order) {
      var cartList = Object.values(order.goods).filter(function (it) { return it.count > 0; });
      if ($('waiting-goods')) $('waiting-goods').innerHTML = cartList.map(function (it) { return '<p class="mb-1">' + it.name + ' \u00d7 ' + it.count + '</p>'; }).join('');
      if ($('waiting-total')) $('waiting-total').textContent = '\u00a5' + order.total.toFixed(2);
      if ($('verify-code')) $('verify-code').textContent = order.verifyCode;
      var qrContainer = $('verify-qr');
      if (qrContainer) {
        qrContainer.textContent = '';
        if (window.QRCode) { try { new QRCode(qrContainer, { text: order.verifyCode, width: 120, height: 120, colorDark: '#000000', colorLight: '#ffffff', correctLevel: QRCode.CorrectLevel.H }); } catch (e) { qrContainer.textContent = order.verifyCode; } }
        else { qrContainer.textContent = order.verifyCode; }
      }
      updateOrderStatusDisplay(null);
      showPage('waiting-page');
      startCheckOrderStatus();
    }

    // ============ 页面切换 ============
    function showPage(page) {
      var pages = ['order-page', 'waiting-page'];
      for (var i = 0; i < pages.length; i++) { var el = document.getElementById(pages[i]); if (el) el.classList.add('hidden'); }
      var bar = $('cart-bar'); if (bar) bar.classList.add('hidden');
      var tgt = document.getElementById(page); if (tgt) tgt.classList.remove('hidden');
      if (page === 'order-page' && bar) bar.classList.remove('hidden');
    }

    // ============ 订单状态显示（防抖：相同取餐码不重复生成二维码）============
    function updateOrderStatusDisplay(order) {
      var titleEl = $('waiting-title'), statusIconEl = $('status-icon'), statusTextEl = $('status-text');
      var verifySection = $('verify-section'), pickupCodeSection = $('pickup-code-section');
      var pickupCodeDisplay = $('pickup-code-display'), pickupQr = $('pickup-qr');
      function renderQr(code) {
        if (!pickupQr) return;
        if (_lastPickupCode === code) return;
        _lastPickupCode = code;
        pickupQr.textContent = '';
        if (window.QRCode) { try { new QRCode(pickupQr, { text: code, width: 120, height: 120, colorDark: '#000000', colorLight: '#ffffff', correctLevel: QRCode.CorrectLevel.H }); } catch (e) { pickupQr.textContent = code; } }
        else { pickupQr.textContent = code; }
      }
      if (!order) {
        if (titleEl) titleEl.textContent = '\u8bf7\u524d\u5f80\u6536\u94f6\u53f0\u4ed8\u6b3e';
        if (statusIconEl) statusIconEl.className = 'fa-solid fa-spinner fa-spin text-coffee-main';
        if (statusTextEl) statusTextEl.textContent = '\u7b49\u5f85\u4ed8\u6b3e...';
        if (verifySection) verifySection.classList.remove('hidden');
        if (pickupCodeSection) pickupCodeSection.classList.add('hidden');
        _lastPickupCode = null;
        return;
      }
      switch (order.status) {
        case 'making':
          if (titleEl) titleEl.textContent = '\u6b63\u5728\u5236\u4f5c\u4e2d...';
          if (statusIconEl) statusIconEl.className = 'fa-solid fa-fire text-orange-500';
          if (statusTextEl) statusTextEl.textContent = '\u6b63\u5728\u5236\u4f5c\u4e2d...';
          if (verifySection) verifySection.classList.add('hidden');
          if (order.pickupCode && pickupCodeSection) { pickupCodeSection.classList.remove('hidden'); if (pickupCodeDisplay) pickupCodeDisplay.textContent = order.pickupCode; renderQr(order.pickupCode); }
          break;
        case 'wait':
          if (titleEl) titleEl.textContent = '\u8bf7\u53d6\u9910';
          if (statusIconEl) statusIconEl.className = 'fa-solid fa-bell text-green-500';
          if (statusTextEl) statusTextEl.textContent = '\u5f85\u53d6\u9910';
          if (verifySection) verifySection.classList.add('hidden');
          if (order.pickupCode && pickupCodeSection) { pickupCodeSection.classList.remove('hidden'); if (pickupCodeDisplay) pickupCodeDisplay.textContent = order.pickupCode; renderQr(order.pickupCode); }
          break;
        case 'finish':
          if (titleEl) titleEl.textContent = '\u5df2\u53d6\u9910';
          if (statusIconEl) statusIconEl.className = 'fa-solid fa-check-circle text-gray-500';
          if (statusTextEl) statusTextEl.textContent = '\u5df2\u5b8c\u6210';
          if (verifySection) verifySection.classList.add('hidden');
          if (order.pickupCode && pickupCodeSection) { pickupCodeSection.classList.remove('hidden'); if (pickupCodeDisplay) pickupCodeDisplay.textContent = order.pickupCode; renderQr(order.pickupCode); }
          break;
        default:
          if (titleEl) titleEl.textContent = '\u8bf7\u524d\u5f80\u6536\u94f6\u53f0\u4ed8\u6b3e';
          if (statusIconEl) statusIconEl.className = 'fa-solid fa-spinner fa-spin text-coffee-main';
          if (statusTextEl) statusTextEl.textContent = '\u7b49\u5f85\u4ed8\u6b3e...';
          if (verifySection) verifySection.classList.remove('hidden');
          if (pickupCodeSection) pickupCodeSection.classList.add('hidden');
          _lastPickupCode = null;
      }
    }

    // ============ 订单状态轮询（保证只有一个 timer）============
    function startCheckOrderStatus() {
      if (orderPollTimer) { clearInterval(orderPollTimer); orderPollTimer = null; }
      orderPollTimer = setInterval(function () {
        if (currentOrderId === null) return;
        var paid = safeGetLS('customerPaidOrders', []);
        var paidOrder = null;
        for (var i = 0; i < paid.length; i++) { if (paid[i].id === currentOrderId) { paidOrder = paid[i]; break; } }
        if (paidOrder) updateOrderStatusDisplay(paidOrder);
      }, 1500);
    }

    // ============ 返回点单页 ============
    if ($('back-to-order')) {
      $('back-to-order').addEventListener('click', function () {
        currentOrderId = null;
        if (orderPollTimer) { clearInterval(orderPollTimer); orderPollTimer = null; }
        showPage('order-page');
      });
    }

    // ============ 商品数据同步（防抖，避免高频重建）============
    var scheduleGoodsRefresh = debounce(function () {
      var newData = safeGetLS('starCoffeeGoods', DEFAULT_GOODS);
      if (newData.length !== GOODS_DATA.length) {
        GOODS_DATA = newData; GOODS_CACHE = {};
        for (var i = 0; i < GOODS_DATA.length; i++) GOODS_CACHE[GOODS_DATA[i].id] = GOODS_DATA[i];
        renderGoodsList();
      } else {
        var changed = false;
        for (var j = 0; j < newData.length; j++) {
          var old = GOODS_CACHE[newData[j].id];
          if (!old || old.stock !== newData[j].stock || old.price !== newData[j].price || old.name !== newData[j].name) { changed = true; break; }
        }
        if (changed) {
          GOODS_DATA = newData; GOODS_CACHE = {};
          for (var k = 0; k < GOODS_DATA.length; k++) GOODS_CACHE[GOODS_DATA[k].id] = GOODS_DATA[k];
          renderGoodsList();
        }
      }
    }, 500);

    window.addEventListener('storage', function (e) { if (e.key === 'starCoffeeGoods') scheduleGoodsRefresh(); });

    // 页面隐藏时暂停轮询（省电）
    document.addEventListener('visibilitychange', function () {
      if (document.hidden && orderPollTimer) { clearInterval(orderPollTimer); orderPollTimer = null; }
      else if (!document.hidden && currentOrderId) startCheckOrderStatus();
    });

    // ============ 初始化 ============
    renderGoodsList();
    updateCartDisplay();
    (function () {
      var paid = safeGetLS('customerPaidOrders', []);
      if (!paid.length) return;
      var last = paid[paid.length - 1];
      if (Date.now() - last.createTime < 1800000) {
        currentOrderId = last.id;
        var cartList = Object.values(last.goods).filter(function (it) { return it.count > 0; });
        if ($('waiting-goods')) $('waiting-goods').innerHTML = cartList.map(function (it) { return '<p class="mb-1">' + it.name + ' \u00d7 ' + it.count + '</p>'; }).join('');
        if ($('waiting-total')) $('waiting-total').textContent = '\u00a5' + last.total.toFixed(2);
        updateOrderStatusDisplay(last);
        showPage('waiting-page');
        startCheckOrderStatus();
      }
    })();

    window.addEventListener('beforeunload', function () { if (orderPollTimer) clearInterval(orderPollTimer); });
  </script>`;

// 找到第一个 <script> 标签和最后一个 </script> 的位置（在 body 内）
var firstScript = html.indexOf('\n  <script>');
var lastScriptEnd = html.lastIndexOf('  </script>');

if (firstScript !== -1 && lastScriptEnd !== -1 && lastScriptEnd > firstScript) {
  var before = html.substring(0, firstScript + 1); // 保留换行
  var after = html.substring(lastScriptEnd); // 保留  </script> 之后的内容
  var patched = before + newScript + '\n' + after;
  fs.writeFileSync(path, patched, 'utf-8');
  console.log('[OK] customer.html script 块已替换');
} else {
  console.log('[FAIL] 无法定位 script 块。firstScript=' + firstScript + ', lastScriptEnd=' + lastScriptEnd);
  // 退而求其次：直接在 </body> 前插入
}
