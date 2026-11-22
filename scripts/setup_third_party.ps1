<#
.SYNOPSIS
    一键拉取本项目依赖的第三方官方组件（按固定 commit，保证两人环境完全一致）。

.DESCRIPTION
    为什么需要这个脚本：
      third_party/ 里的代码是**第三方官方仓库**，我们不把它提交进成果仓库
      （避免把别人的代码和许可证混进我们的提交）。
      代价是每个人都要自己拉一次 —— 所以用这个脚本把版本钉死，
      保证 A 和 B 拿到的是**同一份代码**。

    拉取方式：用 git 按 commit SHA 浅拉取，不需要完整历史，快且稳定。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_third_party.ps1
#>

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"

# 组件定义：名称 / 仓库地址 / 固定 commit / 用途
$components = @(
    @{
        Name = "AutoVC"
        Url  = "https://github.com/auspicious3000/autovc.git"
        Sha  = "c1827a60e83e6d26030b8953843b2f7cf2bd36db"
        Note = "Stage 1 模型（Generator）+ 说话人嵌入 D_VECTOR + 官方权重 + 配套声码器封装"
    },
    @{
        Name = "wavenet_vocoder"
        Url  = "https://github.com/r9y9/wavenet_vocoder.git"
        Sha  = "d4333546eccd3b681b7092e4f52160ad7e62dcc9"
        Note = "WaveNet 声码器。★ 必须是这个 commit：最后一个还含 builder.py 的版本"
    },
    @{
        Name = "Real-Time-Voice-Cloning"
        Url  = "https://github.com/CorentinJ/Real-Time-Voice-Cloning.git"
        # ★ 2026-11-22（成员 A）补钉 commit：原来留空 => 脚本会「删掉重拉」，
        #   而云机常常连不上 GitHub（实测 22 端口直接超时）=> 那一删就没救了。
        Sha  = "890f3a03187195b9829db2079b75c2ba2ab0405c"
        Note = "GE2E 说话人编码器（论文脚注引用），最终系统使用"
    }
)

function Get-CurrentSha([string]$path) {
    if (-not (Test-Path (Join-Path $path ".git"))) { return "" }
    try {
        return (git -C $path rev-parse HEAD 2>$null).Trim()
    } catch {
        return ""
    }
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "  第三方组件一键搭建"
Write-Host "=============================================================="
Write-Host "  目标目录: $thirdParty"
Write-Host ""

New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

foreach ($component in $components) {
    $name = $component.Name
    $url  = $component.Url
    $sha  = $component.Sha
    $path = Join-Path $thirdParty $name

    Write-Host "--- $name ---"
    Write-Host "    $url"
    if ($sha) { Write-Host "    commit: $sha" }
    Write-Host "    用途  : $($component.Note)"

    $current = Get-CurrentSha $path
    if ($current -and $sha -and $current -eq $sha) {
        Write-Host "    [跳过] 已是目标版本" -ForegroundColor DarkGray
        continue
    }

    # ★ 2026-11-22（成员 A）：**断网时不删**。
    #   先备份，拉成功再换；拉失败就**保留原有的**并继续。
    $backup = ""
    if (Test-Path $path) {
        Write-Host "    版本不符，尝试拉取目标版本（失败则保留现有副本）..."
        $backup = "$path.bak.$([DateTimeOffset]::Now.ToUnixTimeSeconds())"
        Copy-Item -Recurse -Force $path $backup
        Remove-Item -Recurse -Force $path
    }

    if ($sha) {
        # 按 SHA 浅拉取：不需要完整历史
        git init -q $path
        git -C $path remote add origin $url
        git -C $path fetch -q --depth 1 origin $sha
        git -C $path checkout -q FETCH_HEAD
    } else {
        git clone -q --depth 1 $url $path
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "    [警告] 拉取失败（网络不通？）" -ForegroundColor Yellow
        if ($backup -and (Test-Path $backup)) {
            Write-Host "    [保留] 恢复原有副本：$backup"
            Move-Item -Force $backup $path
        }
        Write-Host "    [跳过] $name（用现有副本继续）" -ForegroundColor Yellow
        continue
    }
    if ($backup -and (Test-Path $backup)) { Remove-Item -Recurse -Force $backup }
    Write-Host "    [OK] 已就绪  $(Get-CurrentSha $path)" -ForegroundColor Green
}

# --- wavenet_vocoder 的 version.py 是打包时生成的，git 里没有，必须补上 ---
Write-Host ""
Write-Host "--- 生成 wavenet_vocoder/version.py ---"
$versionPath = Join-Path $thirdParty "wavenet_vocoder\wavenet_vocoder\version.py"
if (Test-Path $versionPath) {
    Write-Host "    [跳过] 已存在" -ForegroundColor DarkGray
} else {
    # 与官方 setup.py 的 build_py.create_version_file() 完全一致的写法
    $sha = (git -C (Join-Path $thirdParty "wavenet_vocoder") rev-parse HEAD).Trim()
    $version = "0.1.1+" + $sha.Substring(0, 7)
    Set-Content -Path $versionPath -Value "__version__ = '$version'" -Encoding UTF8
    Write-Host "    [OK] 已生成 $versionPath  ($version)" -ForegroundColor Green
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "  完成。接下来："
Write-Host "  1) 把权重下载到 algorithms/voiceprint_protection/weights/"
Write-Host "     autovc.ckpt / 3000000-BL.ckpt / checkpoint_step001000000_ema.pth"
Write-Host "     下载地址见 third_party/README.md"
Write-Host "  2) 验证环境："
Write-Host "     python algorithms/voiceprint_protection/scripts/check_env.py"
Write-Host "     python algorithms/voiceprint_protection/scripts/verify_autovc_mel.py"
Write-Host "=============================================================="
Write-Host ""
