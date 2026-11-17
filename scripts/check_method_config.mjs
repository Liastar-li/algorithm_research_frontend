// 方法配置契约校验 —— 成员 B 主责（任务板 W3-P1 的验收方式）。
//
// 用法：
//   node scripts/check_method_config.mjs [public/methods/<方法ID>.json]
//   node scripts/check_method_config.mjs            # 默认校验本组的方法配置
//
// 为什么需要它：
//   平台前端在运行时才校验方法 JSON（src/lib/contract.js 的 validateManifest），
//   写错了要等到页面上点开才发现。这里把它变成一条命令行检查。
//   注意：npm test 只覆盖平台内置方法与视频模板，**不会**校验我们新加的方法 JSON，
//   所以这个脚本是必须的。
//
// 退出码：0 = 全部通过；1 = 有问题。

import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { validateManifest, validateParameters, defaultsFor, safeUrl } from '../src/lib/contract.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

const target = process.argv[2] ?? 'public/methods/group_n005_voiceprint_protection.json';
const methodPath = resolve(repoRoot, target);

let failed = false;
function check(title, fn) {
  try {
    fn();
    console.log('  [通过] ' + title);
  } catch (error) {
    failed = true;
    console.log('  [失败] ' + title + '\n         -> ' + error.message);
  }
}

let method;
try {
  method = JSON.parse(await readFile(methodPath, 'utf8'));
} catch (error) {
  console.error(`读不到方法配置 ${methodPath}：${error.message}`);
  process.exit(1);
}

console.log(`方法配置契约校验：${method.id}`);
console.log(`文件：${target}\n`);

check('validateManifest 通过整体契约校验', () => {
  const result = validateManifest(method);
  if (result !== method) throw new Error('返回值不是原对象');
});

check('参数默认值合法', () => {
  const errors = validateParameters(method, defaultsFor(method));
  if (Object.keys(errors).length) throw new Error(JSON.stringify(errors));
});

check('默认值与论文/配置一致（alpha = 0.10）', () => {
  const defaults = defaultsFor(method);
  console.log('         默认值: ' + JSON.stringify(defaults));
  if (defaults.alpha !== 0.1) throw new Error(`alpha 默认值应为 0.1，实际 ${defaults.alpha}`);
});

check('样例素材地址安全（相对路径或 HTTP(S)）', () => {
  for (const example of method.examples) {
    for (const item of [example.input, example.output]) safeUrl(item.src);
  }
});

check('接口地址是完整 HTTP(S) 地址', () => {
  if (!/^https?:\/\//.test(method.adapter.endpoint)) throw new Error('endpoint 格式不对');
  safeUrl(method.adapter.endpoint);
});

check('输入与参数字段名不冲突、不使用保留字', () => {
  const names = new Set();
  for (const item of [...method.inputs, ...method.parameters]) {
    if (names.has(item.name)) throw new Error('重名: ' + item.name);
    if (item.name === 'method_id' || item.name === 'parameters') throw new Error('用了保留字: ' + item.name);
    names.add(item.name);
  }
});

check('标识符合规（全小写 /^[a-z][a-z0-9_]{1,63}$/）', () => {
  const re = /^[a-z][a-z0-9_]{1,63}$/;
  for (const item of [...method.inputs, ...method.parameters]) {
    if (!re.test(item.name)) throw new Error('字段名不合规: ' + item.name);
  }
  if (!re.test(method.id)) throw new Error('方法 ID 不合规: ' + method.id);
});

check('样例引用的文件已存在', () => {
  const missing = [];
  for (const example of method.examples) {
    for (const side of ['input', 'output']) {
      const src = example[side].src;
      if (src.startsWith('http')) continue;
      const abs = resolve(repoRoot, 'public', src);
      if (!existsSync(abs) && !missing.includes(abs)) missing.push(abs);
    }
  }
  if (missing.length) {
    throw new Error('缺少样例文件（先跑 scripts/make_sample.py）：\n         ' + missing.join('\n         '));
  }
});

console.log('\n' + (failed ? '结果：未通过' : '结果：全部通过'));
process.exit(failed ? 1 : 0);
