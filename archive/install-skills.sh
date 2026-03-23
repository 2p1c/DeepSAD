#!/usr/bin/env bash
# 本地安装 skills 脚本
# 用法: ./install-skills.sh
# 安装完成后会自动 commit 并 push 到远端

set -e

CONFIG="skills.config.json"

if ! command -v jq &>/dev/null; then
  echo "❌ 需要安装 jq，请运行: brew install jq"
  exit 1
fi

if ! command -v npx &>/dev/null; then
  echo "❌ 需要安装 Node.js / npx"
  exit 1
fi

# 读取 agents，构建数组参数
AGENT_ARGS=()
while IFS= read -r agent; do
  AGENT_ARGS+=("-a" "$agent")
done < <(jq -r '.agents[]' "$CONFIG")

echo "🤖 目标 agents: ${AGENT_ARGS[*]}"

# 安装所有 enabled: true 的 skills
# 注意：使用 fd 3 读取，避免 npx 交互时消耗 stdin 导致循环提前退出
while IFS= read -r skill <&3; do
  NAME=$(echo "$skill" | jq -r '.name')
  SOURCE=$(echo "$skill" | jq -r '.source')

  # 检查 skills-lock.json 中是否已安装该 skill
  if [[ -f "skills-lock.json" ]] && jq -e --arg name "$NAME" '.skills[$name]' skills-lock.json &>/dev/null; then
    echo "⏭️  Skill '$NAME' 已安装，跳过（如需更新请先删除 skills-lock.json 中对应条目）"
    continue
  fi

  echo ""
  echo "📦 Installing skill: $NAME (from $SOURCE)"
  npx skills add "$SOURCE" --skill "$NAME" "${AGENT_ARGS[@]}" -y || echo "⚠️  Skill '$NAME' 安装失败，跳过继续"
done 3< <(jq -c '.skills[] | select(.enabled == true)' "$CONFIG")

echo ""
echo "✅ 所有 skill 安装完成"
