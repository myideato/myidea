# OpenClaw 安装指南

## 方法一：直接安装官方包（推荐）

在 PowerShell 中执行以下命令：

```powershell
npm install -g openclaw@latest
```

安装完成后，运行配置向导：

```powershell
openclaw onboard --install-daemon
```

---

## 方法二：如果网络慢，配置国内镜像源

```powershell
# 设置淘宝镜像源
npm config set registry https://registry.npmmirror.com/

# 验证配置
npm config get registry

# 然后安装
npm install -g openclaw@latest
```

---

## 方法三：使用中文社区版（国内用户友好）

专为国内用户优化的中文社区版，包名为 `openclaw-cn`：已完成中文本地化，依赖预构建，在国内网络下安装更顺畅。

```powershell
npm install -g openclaw-cn@latest
```

安装后使用的命令是 `openclaw-cn`：

```powershell
openclaw-cn onboard --install-daemon
```

---

## 验证安装

安装成功后，运行：

```powershell
openclaw --version
```

若能显示版本号（如 `2026.x.x`），说明安装成功。

> 若使用方法三（`openclaw-cn`），可将上述验证命令中的 `openclaw` 替换为 `openclaw-cn` 进行版本检查。

---

## 启动服务

```powershell
# 启动网关
openclaw gateway start

# 查看状态
openclaw status

# 打开控制面板
openclaw dashboard
```

若日常使用 `openclaw-cn`，请将命令前缀改为 `openclaw-cn`（例如 `openclaw-cn gateway start`）。
