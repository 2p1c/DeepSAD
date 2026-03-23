## awesome-code

一个面向 AI 开发者的 GitHub 项目模板，用来集中管理项目级 Skills，并让它们在 `VS Code`、`Claude Code`、`Antigravity` 等工具中复用。

你只需要维护 `skills.config.json`，再运行本地安装脚本，就能按需安装当前项目需要的 Skills。

## 适用场景

- 为 AI 编程项目维护统一的 Skill 清单
- 在多个 Agent / IDE 工具之间复用同一套能力
- 通过配置文件启用或关闭指定 Skill
- 将 Skill 安装状态纳入仓库管理，方便团队协作

## 当前特性

- 使用 `skills.config.json` 声明目标 Agent 与 Skill 列表
- 仅安装 `enabled: true` 的 Skill
- 基于 `skills-lock.json` 跳过已安装 Skill，避免重复安装
- 通过 `npx skills add` 执行实际安装
- 支持从不同来源安装 Skill，例如 GitHub 仓库或第三方 Skill 集合

## 项目结构

```text
.
├── README.md
├── install-skills.sh
├── skills.config.json
├── skills-lock.json
├── copilot-instructions.md
├── CLAUDE.md
└── .agents/
    └── skills/
```

## 环境要求

在运行安装脚本前，请确保本机已安装：

- `Node.js` 与 `npx`
- `jq`

如果你使用 macOS，可以先执行：

```zsh
brew install jq
```

## 快速开始

1. 克隆仓库
2. 按需修改 `skills.config.json`
3. 运行安装脚本

```zsh
git clone <your-repo-url>
cd awesome-code
./install-skills.sh
```

脚本会：

1. 读取 `agents`
2. 找出所有 `enabled: true` 的 Skill
3. 检查 `skills-lock.json`，跳过已安装项
4. 调用 `npx skills add` 执行安装

## 配置说明

### `skills.config.json`

这是项目的主配置文件，包含两部分：

#### `agents`

声明当前项目希望接入的 Agent 工具，例如：

```json
{
	"agents": ["claude-code", "github-copilot", "antigravity"]
}
```

#### `skills`

每个 Skill 包含以下字段：

- `name`: Skill 名称
- `source`: Skill 来源
- `enabled`: 是否启用安装
- `description`: Skill 说明

示例：

```json
{
	"name": "git-commit",
	"source": "github/awesome-copilot",
	"enabled": true,
	"description": "根据当前修改自动总结 git commit message 并提交"
}
```

## 已配置 Skill 示例

当前仓库中的配置示例包括：

- `skill-creator`：创建和优化自定义 Skill
- `find-skills`：查找和管理 Skill
- `git-flow-branch-creator`：创建 Git Flow 分支
- `git-commit`：自动生成并执行规范化提交
- `gws-gmail`：Gmail 邮件查询与管理

说明：`skills.config.json` 表示“期望安装的配置”，`skills-lock.json` 表示“当前已记录安装结果”。如果某个 Skill 在配置中启用但尚未出现在锁文件中，通常表示它还没成功安装或尚未写入锁文件。

## `skills-lock.json` 的作用

`skills-lock.json` 用于记录已经安装过的 Skill 及其来源哈希，主要用途：

- 避免重复安装
- 固定当前 Skill 来源状态
- 便于团队成员同步安装结果

通常建议将它提交到仓库中统一管理。

## 安装脚本行为说明

`install-skills.sh` 当前已经实现：

- 校验 `jq` 是否存在
- 校验 `npx` 是否存在
- 为每个 Agent 组装安装参数
- 逐个安装启用的 Skill
- 在单个 Skill 安装失败时继续处理后续项目

当前脚本**不会自动执行** Git `commit` 或 `push`。如果你希望在安装后提交变更，需要手动执行：

```zsh
git status
git add .
git commit -m "chore: update installed skills"
git push
```

## 常见工作流

### 新增一个 Skill

1. 在 `skills.config.json` 中添加 Skill 条目
2. 将 `enabled` 设为 `true`
3. 运行安装脚本
4. 检查 `.agents/skills/` 和 `skills-lock.json` 是否更新

### 禁用一个 Skill

1. 在 `skills.config.json` 中将 `enabled` 改为 `false`
2. 如有需要，手动清理本地对应 Skill 目录
3. 提交配置变更

## 注意事项

- `skills-lock.json` 中存在的 Skill 会被安装脚本跳过
- 安装失败的 Skill 不会中断整个安装流程
- 不同 Skill 的运行时依赖可能不同，是否可用取决于对应 Skill 自身要求
- 若要重新安装某个已记录的 Skill，可先删除 `skills-lock.json` 中对应条目后再运行脚本

## 适合团队如何使用

如果你在团队中维护 AI 编程工作流，推荐采用以下约定：

- 用 `skills.config.json` 维护“目标能力集”
- 用 `skills-lock.json` 维护“当前安装状态”
- 将常用 Agent 说明写入 `CLAUDE.md`、`copilot-instructions.md` 等文件
- 通过 Pull Request 审核新增 Skill 的来源、描述与启用状态

## 后续可扩展方向

- 为 `install-skills.sh` 增加 `--force` 选项，支持强制重装
- 增加安装结果校验与失败汇总
- 增加自动同步 `skills-lock.json` 的检查逻辑
- 增加 CI 校验，确保配置文件和锁文件一致

