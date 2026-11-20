// VCTK 定向下载器 —— 只取需要的部分，并行 range 请求。
//
// 【为什么需要它】
// VCTK 官方压缩包 VCTK-Corpus-0.92.zip 有 11.75 GB，但里面我们只用到一小半：
//   wav48_silence_trimmed/   48 kHz FLAC（已裁静音）  ← 要这个
//   wav/                     48 kHz WAV（未裁静音）      ← 用不上（是前者的大 1.5 倍）
//   txt/                     文本标注                    ← 用不上
// 而从国内连 Edinburgh 的服务器，单连接只有 ~30 KB/s（16 连接 ~280 KB/s，48 连接 ~420 KB/s），
// 整包下完要 8~11 小时。
//
// 这个脚本做两件事：
//   1) **只下载需要的条目**（默认只要 wav48_silence_trimmed/），数据量直接减半
//   2) **并发 range 请求**（默认 48），比单连接快一个数量级
//
// 它不下载 zip 容器本身，而是**直接按条目解压落盘** —— 省掉最后再解一次 zip 的时间。
//
// 【用法】
//   node scripts/download_vctk.mjs --out D:\vpp\data\raw
//   node scripts/download_vctk.mjs --out D:\vpp\data\raw --concurrency 64
//   node scripts/download_vctk.mjs --out D:\vpp\data\raw --include wav48_silence_trimmed --include txt
//   node scripts/download_vctk.mjs --out D:\vpp\data\raw --dry-run     # 只看清单不下载
//   node scripts/download_vctk.mjs --out D:\vpp\data\raw --limit 20     # 只下前 20 个条目（试跑）
//
// 【断点续传】
// 已经落盘且大小正确的文件会跳过，所以中断后重跑同一条命令即可。
//
// 【依赖】只用 Node 内置模块（fetch / zlib），无需 npm install。

import { createWriteStream } from 'node:fs';
import { mkdir, stat, writeFile, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { inflateRawSync } from 'node:zlib';
import { pipeline } from 'node:stream/promises';

const DEFAULT_URL =
  'https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content';

const args = process.argv.slice(2);
function arg(name, fallback) {
  const i = args.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const next = args[i + 1];
  return next && !next.startsWith('--') ? next : true;
}

const URL = arg('url', DEFAULT_URL);
const OUT = resolve(String(arg('out', 'D:/vpp/data/raw')));
const CONCURRENCY = Number(arg('concurrency', 48));
const INCLUDES = args.includes('--include')
  ? args.reduce((acc, cur, i) => (cur === '--include' ? [...acc, args[i + 1]] : acc), [])
  : ['wav48_silence_trimmed'];
const LIMIT = Number(arg('limit', 0));
const DRY_RUN = !!arg('dry-run', false);
// 每个说话人最多取几条（0 = 全部）。取值均匀分布，避免只拿到前几句造成偏差。
const PER_SPEAKER = Number(arg('per-speaker', 0));
// 只下这些说话人（逗号分隔，如 p225,p292）；不给就全要
const SPEAKERS = arg('speakers', null)
  ? String(arg('speakers')).split(',').map((s) => s.trim()).filter(Boolean)
  : null;

const MB = 1024 * 1024;
const GB = 1024 * MB;

// ---------------------------------------------------------------------------
// HTTP range 辅助
// ---------------------------------------------------------------------------
let TOTAL = 0;

async function head() {
  const response = await fetch(URL, { method: 'HEAD' });
  if (!response.ok) throw new Error(`HEAD 失败：HTTP ${response.status}`);
  const length = Number(response.headers.get('content-length'));
  if (!Number.isFinite(length)) throw new Error('拿不到 content-length');
  return { size: length, ranges: response.headers.get('accept-ranges') };
}

async function getRange(start, end, retries = 4) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetch(URL, { headers: { Range: `bytes=${start}-${end}` } });
      if (response.status === 206) return Buffer.from(await response.arrayBuffer());
      if (response.status === 200 && start === 0 && end >= TOTAL - 1) {
        // 服务器忽略了 Range，直接返回整包（不太可能，但要兜住）
        return Buffer.from(await response.arrayBuffer());
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      if (attempt === retries - 1) throw error;
      await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
    }
  }
}

/** 把 [start,end] 切成若干块并发拉取再拼接（中央目录有好几 MB，单连接太慢） */
async function getRangeParallel(start, end, workers = 16) {
  const length = end - start + 1;
  if (length < 4 * MB) return getRange(start, end);
  const chunk = Math.ceil(length / workers);
  const parts = [];
  for (let i = 0; i < workers; i++) {
    const s = start + i * chunk;
    const e = Math.min(end, s + chunk - 1);
    if (s > end) break;
    parts.push(getRange(s, e));
  }
  const buffers = await Promise.all(parts);
  return Buffer.concat(buffers);
}

// ---------------------------------------------------------------------------
// zip 中央目录解析（含 ZIP64）
// ---------------------------------------------------------------------------
async function readCentralDirectory() {
  const tailLen = Math.min(65536, TOTAL);
  const tail = await getRange(TOTAL - tailLen, TOTAL - 1);

  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('不是有效的 zip（找不到 EOCD）');

  let entries = tail.readUInt16LE(eocd + 10);
  let cdSize = tail.readUInt32LE(eocd + 12);
  let cdOffset = tail.readUInt32LE(eocd + 16);

  if (cdSize === 0xffffffff || cdOffset === 0xffffffff || entries === 0xffff) {
    let loc = -1;
    for (let i = eocd - 20; i >= 0; i--) {
      if (tail.readUInt32LE(i) === 0x07064b50) { loc = i; break; }
    }
    if (loc < 0) throw new Error('需要 ZIP64 但找不到 locator');
    const z64Offset = Number(tail.readBigUInt64LE(loc + 8));
    const z64 = await getRange(z64Offset, z64Offset + 55);
    if (z64.readUInt32LE(0) !== 0x06064b50) throw new Error('ZIP64 EOCD 签名不对');
    entries = Number(z64.readBigUInt64LE(32));
    cdSize = Number(z64.readBigUInt64LE(40));
    cdOffset = Number(z64.readBigUInt64LE(48));
  }
  return { entries, cdSize, cdOffset };
}

function parseCentralDirectory(cd) {
  const list = [];
  let p = 0;
  while (p + 46 <= cd.length && cd.readUInt32LE(p) === 0x02014b50) {
    const method = cd.readUInt16LE(p + 10);
    let compSize = cd.readUInt32LE(p + 20);
    let uncompSize = cd.readUInt32LE(p + 24);
    const nameLen = cd.readUInt16LE(p + 28);
    const extraLen = cd.readUInt16LE(p + 30);
    let localOffset = cd.readUInt32LE(p + 42);
    const name = cd.toString('utf8', p + 46, p + 46 + nameLen);

    if (uncompSize === 0xffffffff || compSize === 0xffffffff || localOffset === 0xffffffff) {
      let e = p + 46 + nameLen;
      const end = e + extraLen;
      while (e + 4 <= end) {
        const id = cd.readUInt16LE(e);
        const len = cd.readUInt16LE(e + 2);
        if (id === 0x0001) {
          let q = e + 4;
          if (uncompSize === 0xffffffff) { uncompSize = Number(cd.readBigUInt64LE(q)); q += 8; }
          if (compSize === 0xffffffff) { compSize = Number(cd.readBigUInt64LE(q)); q += 8; }
          if (localOffset === 0xffffffff) { localOffset = Number(cd.readBigUInt64LE(q)); q += 8; }
        }
        e += 4 + len;
      }
    }
    if (!name.endsWith('/')) {
      list.push({ name, method, compSize, uncompSize, localOffset });
    }
    p += 46 + nameLen + extraLen;
  }
  return list;
}

// ---------------------------------------------------------------------------
// 下载
// ---------------------------------------------------------------------------
async function downloadEntry(entry) {
  const target = join(OUT, entry.name.replaceAll('/', require$sep()));
  try {
    const info = await stat(target);
    if (info.size === entry.uncompSize) return { skipped: true };
  } catch { /* 不存在，正常 */ }

  // 一次请求拿到：本地文件头 + 数据 + 一点点冗余（用于解析本地头里的 extra 长度）
  const probeExtra = 512;
  const headerEnd = entry.localOffset + 30 + 255 + probeExtra;
  const buffer = await getRange(entry.localOffset, Math.min(TOTAL - 1, headerEnd));

  if (buffer.readUInt32LE(0) !== 0x04034b50) throw new Error(`本地文件头签名不对：${entry.name}`);
  const nameLen = buffer.readUInt16LE(26);
  const extraLen = buffer.readUInt16LE(28);
  const dataStart = 30 + nameLen + extraLen;
  const dataEnd = dataStart + entry.compSize;
  if (dataEnd > buffer.length) {
    // 冗余不够，重新按精确长度取
    const exact = await getRange(entry.localOffset, entry.localOffset + dataEnd - 1);
    return writeEntry(exact, entry, target);
  }
  return writeEntry(buffer, entry, target);
}

const require$sep = () => (process.platform === 'win32' ? '\\' : '/');

async function writeEntry(buffer, entry, target) {
  const nameLen = buffer.readUInt16LE(26);
  const extraLen = buffer.readUInt16LE(28);
  const dataStart = 30 + nameLen + extraLen;
  const raw = buffer.subarray(dataStart, dataStart + entry.compSize);

  let content;
  if (entry.method === 0) content = raw;
  else if (entry.method === 8) content = inflateRawSync(raw);
  else throw new Error(`不支持的压缩方式 ${entry.method}：${entry.name}`);

  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, content);
  return { bytes: content.length };
}

// ---------------------------------------------------------------------------
// 并发调度
// ---------------------------------------------------------------------------
async function runPool(items, worker, limit, totalBytes) {
  let index = 0;
  let done = 0;
  let bytes = 0;
  const started = Date.now();
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const current = index++;
      if (current >= items.length) return;
      const item = items[current];
      try {
        const result = await worker(item);
        if (result?.bytes) bytes += result.bytes;
      } catch (error) {
        console.error(`  [失败] ${item.name}：${error.message}`);
      }
      done++;
      if (done % 100 === 0 || done === items.length) {
        const seconds = (Date.now() - started) / 1000;
        const rate = bytes / 1024 / Math.max(seconds, 0.001);
        const eta = rate > 0 ? (totalBytes - bytes) / 1024 / rate : 0;
        console.log(
          `  ${done}/${items.length}  ${(bytes / GB).toFixed(2)} GiB  ${rate.toFixed(0)} KiB/s  ETA ${(eta / 3600).toFixed(1)}h`
        );
      }
    }
  });
  await Promise.all(workers);
}

// ---------------------------------------------------------------------------
async function main() {
  console.log('='.repeat(64));
  console.log('  VCTK 定向下载器（只取需要的目录 + 并发 range）');
  console.log('='.repeat(64));

  const info = await head();
  TOTAL = info.size;
  console.log(`\n  源      : ${URL}`);
  console.log(`  总大小  : ${(TOTAL / GB).toFixed(2)} GiB (${TOTAL} 字节)`);
  console.log(`  Range   : ${info.ranges || '（未知）'}`);
  console.log(`  输出目录: ${OUT}`);
  console.log(`  只要    : ${INCLUDES.join(', ')}`);
  console.log(`  并发数  : ${CONCURRENCY}`);

  console.log('\n  读取中央目录…');
  const cachePath = join(OUT, '.vctk_entries.json');
  let all = null;
  try {
    const cached = JSON.parse(await readFile(cachePath, 'utf8'));
    if (cached.total === TOTAL && cached.entries?.length) {
      all = cached.entries;
      console.log(`  用本地缓存的中央目录（${all.length} 个条目），跳过下载`);
    }
  } catch { /* 没有缓存，正常 */ }

  if (!all) {
    const cd = await readCentralDirectory();
    console.log(`  中央目录：${cd.entries} 个条目，${(cd.cdSize / MB).toFixed(1)} MB`);
    const cdBuffer = await getRangeParallel(cd.cdOffset, cd.cdOffset + cd.cdSize - 1, 16);
    all = parseCentralDirectory(cdBuffer);
    console.log(`  解析到 ${all.length} 个文件条目`);
    try {
      await mkdir(OUT, { recursive: true });
      await writeFile(cachePath, JSON.stringify({ total: TOTAL, entries: all }));
      console.log(`  已缓存到 ${cachePath}`);
    } catch (error) {
      console.log(`  [提示] 缓存写入失败（不影响下载）：${error.message}`);
    }
  }

  // 按顶层目录统计，看看哪些值得下
  const groups = new Map();
  const speakers = new Set();
  for (const e of all) {
    const top = e.name.split('/')[0];
    const g = groups.get(top) || { count: 0, comp: 0, uncomp: 0 };
    g.count++; g.comp += e.compSize; g.uncomp += e.uncompSize;
    groups.set(top, g);
    if (top === 'wav48_silence_trimmed') speakers.add(e.name.split('/')[1]);
  }
  console.log('\n  【按顶层目录】');
  for (const [name, g] of [...groups.entries()].sort((a, b) => b[1].comp - a[1].comp)) {
    console.log(`    ${name.padEnd(28)} ${String(g.count).padStart(7)} 个  压缩 ${(g.comp / GB).toFixed(3)} GiB  原样 ${(g.uncomp / GB).toFixed(3)} GiB`);
  }
  console.log(`    （wav48_silence_trimmed 下共有 ${speakers.size} 个说话人目录）`);

  // 按"目录"分组（wav48_silence_trimmed/<说话人>/<文件>）
  const bySpeaker = new Map();
  const extras = [];
  for (const e of all) {
    // 顶层的小元数据文件（speaker-info.txt / update.txt）一并带上：
    // 它们记录了录音配置，写 README 的"数据来源"一节要用
    if (!e.name.includes('/')) {
      if (e.name.endsWith('.txt') && e.compSize < 1024 * 1024) extras.push(e);
      continue;
    }
    if (!INCLUDES.some((inc) => e.name.includes(inc))) continue;
    const parts = e.name.split('/');
    const speaker = parts[1] ?? parts[0];
    if (SPEAKERS && !SPEAKERS.includes(speaker)) continue;
    if (!bySpeaker.has(speaker)) bySpeaker.set(speaker, []);
    bySpeaker.get(speaker).push(e);
  }

  const AUDIO_EXTS = ['.flac', '.wav', '.mp3'];
  let wanted = [];
  for (const speaker of [...bySpeaker.keys()].sort()) {
    const list = bySpeaker.get(speaker);
    // ★ 用**码点序**排序，不要用 localeCompare：
    //   localeCompare 依赖运行机器的 locale 与 ICU 版本，
    //   两台机器可能挑出**不同的**句子 -> 数据不可复现。
    list.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    // ★ 配额只算**音频文件**：wav48_silence_trimmed/ 里混着一个 log.txt，
    //   它不是音频，不该占掉一个名额（prepare_data 也只认 .flac/.wav）；
    //   但它要一起下载（那是 VCTK 的日志，README 要引）。
    const isAudio = (e) => AUDIO_EXTS.some((ext) => e.name.toLowerCase().endsWith(ext));
    const audio = list.filter(isAudio);
    const others = list.filter((e) => !isAudio(e));
    if (PER_SPEAKER > 0 && audio.length > PER_SPEAKER) {
      // 均匀取样：覆盖整句话表，而不是只取开头几句
      const picked = [];
      for (let i = 0; i < PER_SPEAKER; i++) {
        picked.push(audio[Math.floor((i * audio.length) / PER_SPEAKER)]);
      }
      wanted.push(...picked);
    } else {
      wanted.push(...audio);
    }
    wanted.push(...others);
  }

  const totalBytes = wanted.reduce((sum, e) => sum + e.compSize, 0);
  console.log(`\n  说话人 ${bySpeaker.size} 个`
    + (PER_SPEAKER ? `，每人最多 ${PER_SPEAKER} 句` : '，每人全部')
    + (SPEAKERS ? `（指定：${SPEAKERS.join(',')}）` : ''));
  console.log(`  选中 ${wanted.length} 个条目，共 ${(totalBytes / GB).toFixed(2)} GiB`
    + `（整包的 ${((totalBytes / TOTAL) * 100).toFixed(0)}%）`
    + (extras.length ? `，另含 ${extras.length} 个元数据文件` : ''));

  if (DRY_RUN) {
    console.log('\n  [干跑] 没有下载任何东西。');
    return;
  }
  if (!wanted.length) {
    console.log('\n  [失败] 没有匹配的条目，检查 --include 的写法。');
    return;
  }

  const list = LIMIT > 0 ? wanted.slice(0, LIMIT) : [...wanted, ...extras];
  console.log(`\n  开始下载（${list.length} 个条目，并发 ${CONCURRENCY}）…\n`);
  const started = Date.now();
  await runPool(list, downloadEntry, CONCURRENCY, totalBytes);
  const seconds = (Date.now() - started) / 1000;
  console.log(`\n  完成，用时 ${(seconds / 60).toFixed(1)} 分钟。`);
  console.log(`  输出目录：${OUT}`);
  console.log('  接下来：');
  console.log('    python algorithms/voiceprint_protection/scripts/survey_vctk.py');
  console.log('    python algorithms/voiceprint_protection/scripts/prepare_data.py');
}

main().catch((error) => {
  console.error('\n[失败]', error.message);
  process.exit(1);
});
