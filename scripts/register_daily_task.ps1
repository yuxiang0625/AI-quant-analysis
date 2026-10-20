<#
.SYNOPSIS
    注册 / 注销 Aurum 每日数据更新的 Windows 计划任务。

.DESCRIPTION
    本脚本只做计划任务的注册与注销，不直接抓取数据。
    注册后由 Windows 任务计划程序按设定时间调用：
        python -m scripts.daily_update --commit --quiet

    任务特性：
      - 工作目录固定为项目根目录，确保相对路径与 .env 能被正确读取
      - 失败自动重试 3 次，间隔 10 分钟（网络临时故障可自愈）
      - 最长执行 1 小时后强制结束，避免任务卡死堆积
      - 仅在接通电源时运行，笔记本用电池时不触发
      - 输出重定向到 data\daily_update\ 下的日志文件

.NOTES
    注册计划任务会修改系统状态。默认注册到「当前用户」下，
    不需要管理员权限；若要以 SYSTEM 账户运行需自行改用 -RunLevel Highest。
#>

[CmdletBinding()]
param(
    # 任务名称
    [string]$TaskName = "Aurum Daily Data Update",

    # 每日触发时间（24 小时制）
    [string]$Time = "08:30",

    # 动作：register 注册 / unregister 注销 / show 查看
    [ValidateSet("register", "unregister", "show")]
    [string]$Action = "register",

    # 项目根目录，默认取本脚本的上级目录
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    # Python 解释器路径，默认取当前 PATH 中的 python
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------
# 1. 解析 Python 解释器路径
# ------------------------------------------------------------------

if (-not $PythonExe) {
    $found = Get-Command python -ErrorAction SilentlyContinue

    if (-not $found) {
        throw "找不到 python，请用 -PythonExe 指定解释器完整路径。"
    }

    $PythonExe = $found.Source
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 解释器不存在：$PythonExe"
}

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "项目根目录不存在：$ProjectRoot"
}

Write-Host "项目根目录 : $ProjectRoot"
Write-Host "Python     : $PythonExe"
Write-Host "任务名称   : $TaskName"
Write-Host ""

# ------------------------------------------------------------------
# 2. 查看
# ------------------------------------------------------------------

if ($Action -eq "show") {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

    if (-not $task) {
        Write-Host "该计划任务尚未注册。"
        return
    }

    Write-Host "状态       : $($task.State)"
    Write-Host "触发器     :"
    $task.Triggers | ForEach-Object { Write-Host "             $($_.StartBoundary)  (每日)" }
    Write-Host "执行命令   : $($task.Actions[0].Execute) $($task.Actions[0].Arguments)"
    Write-Host "工作目录   : $($task.Actions[0].WorkingDirectory)"
    Write-Host ""
    Write-Host "最近运行结果："
    Get-ScheduledTaskInfo -TaskName $TaskName |
        Select-Object LastRunTime, LastTaskResult, NextRunTime, NumberOfMissedRuns |
        Format-List
    return
}

# ------------------------------------------------------------------
# 3. 注销
# ------------------------------------------------------------------

if ($Action -eq "unregister") {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

    if (-not $task) {
        Write-Host "该计划任务不存在，无需注销。"
        return
    }

    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false

    Write-Host "已注销计划任务：$TaskName"
    return
}

# ------------------------------------------------------------------
# 4. 注册
# ------------------------------------------------------------------

$logDir = Join-Path $ProjectRoot "data\daily_update"

if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    Write-Host "已创建日志目录：$logDir"
}

$stdoutLog = Join-Path $logDir "task_stdout.log"
$stderrLog = Join-Path $logDir "task_stderr.log"

# 用 cmd 包一层以便重定向输出；任务本身仍由 python 执行
$innerCommand = "`"$PythonExe`" -m scripts.daily_update --commit --quiet 1>> `"$stdoutLog`" 2>> `"$stderrLog`""

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c $innerCommand" `
    -WorkingDirectory $ProjectRoot

# 每日触发
$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries:$false `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Limited

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existing) {
    Write-Host "计划任务已存在，执行更新。"

    Set-ScheduledTask -TaskName $TaskName `
        -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal | Out-Null
}
else {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Aurum Intelligence 每日数据更新：黄金行情、宏观数据、COMEX 黄金库存" | Out-Null

    Write-Host "已注册计划任务。"
}

Write-Host ""
Write-Host "完成。每日 $Time 自动执行，失败重试 3 次。"
Write-Host ""
Write-Host "常用操作："
Write-Host "  立即手动跑一次 : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  查看运行状态   : powershell -File scripts\register_daily_task.ps1 -Action show"
Write-Host "  注销任务       : powershell -File scripts\register_daily_task.ps1 -Action unregister"
Write-Host "  查看日志       : $stdoutLog"
