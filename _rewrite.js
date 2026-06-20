const fs = require('fs');
const h = fs.readFileSync('/workspace/index.html', 'utf-8');

// ==== 替换 JS ====
const jsA = h.indexOf('    // ☕ AI 语音助手');
if (jsA < 0) { console.error('JS 起点缺失'); process.exit(1); }
const tmp = h.substring(jsA);
const iifeRel = tmp.indexOf('    })();');
const jsB = jsA + iifeRel + '    })();'.length;
const newJS = fs.readFileSync('/workspace/_new_ai_js.txt', 'utf-8');
let out = h.substring(0, jsA) + newJS + h.substring(jsB);
console.log('JS 替换完成');

// ==== 替换 HTML（底部）====
const htmlA = out.indexOf('  <!-- ======== AI 语音助手（简洁版）');
if (htmlA < 0) { console.error('HTML 起点缺失'); process.exit(1); }
const bodyEnd = out.indexOf('</body>', htmlA);
const newHTML = fs.readFileSync('/workspace/_new_ai_html.txt', 'utf-8');
out = out.substring(0, htmlA) + newHTML + out.substring(bodyEnd);

fs.writeFileSync('/workspace/index.html', out);
console.log('OK - final length:', out.length, 'lines:', out.split('\n').length);
