# Copilot Instructions

## 🎯 核心协作原则

### 理解优先工作流

- **理解万岁**: 当接收到需要修改代码的指令后,总是先分析完成指令所需的信息或者对实现方式的理解,将自己的理解用简明扼要的语言向用户表述或提出问题,用户回答问题或确认信息后再开始编写代码,以此确保理解用户意图。

- **告诉过你**: 如果认为用户没有提供足够的信息,或者指令不明确,请礼貌地要求用户提供更多信息或澄清指令。

- **暂时安全**: 在确认信息或提出问题后,请耐心等待用户回复,当接收到用户一次回复后,可以开始修改代码,如果缺少重要信息,可以再次提出问题,并确保在用户回复后再继续执行下一步操作。

- **不要猜测**: 在没有用户明确指示的情况下,不要猜测用户的需求,请始终等待用户明确指示后再开始编写代码。

- **最强大脑**：在原有基础上添加一条额外的思考逻辑来辅助你理解和决策，那就是想一想“该领域的最强大脑会怎么做？”。

### 沟通示例

```
用户: "实现一个卷积神经网络"

AI 应该回应:
"我理解您想实现一个CNN。为了提供最合适的实现,我需要确认:
1. 使用什么框架?(PyTorch / TensorFlow / JAX)
2. 任务类型是什么?(图像分类 / 目标检测 / 分割)
3. 输入数据的维度?
4. 是否有特定的网络结构要求?(如ResNet / VGG风格)
5. 是否需要预训练权重?

请告诉我这些信息,我会据此设计最适合的架构。"
```

## 💻 代码质量标准

### 通用原则

- **可读性第一**: 清晰胜过简洁,明确胜过巧妙
- **注释到位**: 解释"为什么"这样做,而不仅是"做了什么"
- **错误处理**: 任何可能失败的操作都要有适当的错误处理
- **模块化设计**: 功能独立、职责单一、易于测试
- **文档完整**: 每个函数/类都有清晰的文档说明

# Copilot Instructions for Web Development

## 项目概述

这是一个由新手开发者维护的网站项目。请提供清晰、易懂的代码建议,并附带详细注释。

## 核心原则

### 1. 代码质量

- **简洁明了**: 优先考虑可读性,避免过度复杂的实现
- **详细注释**: 为关键逻辑添加中文注释,解释"为什么"而不仅是"是什么"
- **渐进式开发**: 从简单实现开始,逐步优化
- **错误处理**: 始终包含适当的错误处理和验证

### 2. 技术栈偏好

- **HTML**: 使用语义化标签,确保良好的可访问性
- **CSS**: 优先使用现代 CSS 特性(Flexbox, Grid),避免过时的方法
- **JavaScript**: 使用 ES6+语法,优先使用 const/let 而非 var
- **框架**: 如使用框架,请明确说明依赖项

## 编码规范

### HTML

```html
<!-- ✅ 推荐: 语义化、结构清晰 -->
<header>
  <nav aria-label="主导航">
    <ul>
      <li><a href="#home">首页</a></li>
    </ul>
  </nav>
</header>

<!-- ❌ 避免: 过度使用div -->
<div class="header">
  <div class="nav">...</div>
</div>
```

### CSS

```css
/* ✅ 推荐: 使用CSS变量和现代布局 */
:root {
  --primary-color: #3498db;
  --spacing: 1rem;
}

.container {
  display: grid;
  gap: var(--spacing);
}

/* ❌ 避免: 过时的浮动布局 */
.old-style {
  float: left;
  clear: both;
}
```

### JavaScript

```javascript
// ✅ 推荐: 现代语法,清晰的错误处理
const fetchData = async (url) => {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("获取数据失败:", error);
    return null;
  }
};

// ❌ 避免: 回调地狱,缺少错误处理
function getData(url, callback) {
  fetch(url)
    .then((r) => r.json())
    .then(callback);
}
```

## 常见错误预防

### 1. 安全性

- **XSS 防护**: 始终清理用户输入
- **CSRF 保护**: 敏感操作需要令牌验证
- **密码存储**: 永远不要在前端存储敏感信息
- **API 密钥**: 使用环境变量,不要硬编码

```javascript
// ✅ 安全地插入用户内容
element.textContent = userInput; // 自动转义

// ❌ 危险的做法
element.innerHTML = userInput; // 可能导致XSS攻击
```

### 2. 性能优化

- **图片优化**: 使用适当的格式(WebP)和尺寸
- **懒加载**: 对非关键资源使用懒加载
- **代码分割**: 避免单一巨大的 JS 文件
- **缓存策略**: 合理使用浏览器缓存

```javascript
// ✅ 使用防抖处理高频事件
const debounce = (fn, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
};

window.addEventListener("resize", debounce(handleResize, 300));
```

### 3. 响应式设计

- 移动优先的设计方法
- 使用相对单位(rem, %, vw/vh)
- 测试多种屏幕尺寸

```css
/* ✅ 移动优先的响应式设计 */
.container {
  width: 100%;
  padding: 1rem;
}

@media (min-width: 768px) {
  .container {
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

## 代码建议格式

当提供代码建议时,请遵循以下格式:

```
1. **简短说明**: 这段代码做什么
2. **代码实现**: 完整的、可运行的代码
3. **关键注释**: 解释重要逻辑
4. **注意事项**: 可能的陷阱或需要注意的地方
```

## 文件组织

```
project/
├── index.html          # 主页面
├── css/
│   ├── style.css      # 主样式
│   └── responsive.css # 响应式样式
├── js/
│   ├── main.js        # 主逻辑
│   └── utils.js       # 工具函数
├── assets/
│   ├── images/        # 图片资源
│   └── fonts/         # 字体文件
└── README.md          # 项目说明
```

## 调试技巧

```javascript
// ✅ 使用有意义的console信息
console.log("用户数据:", { userId, userName, timestamp: Date.now() });

// ✅ 使用断言检查假设
console.assert(user.age >= 0, "年龄不能为负数");

// ❌ 避免无意义的日志
console.log("1");
console.log("here");
```

## 可访问性(A11y)清单

- [ ] 所有图片都有 alt 属性
- [ ] 表单输入有对应的 label
- [ ] 使用语义化 HTML 标签
- [ ] 键盘可以访问所有交互元素
- [ ] 颜色对比度符合 WCAG 标准
- [ ] 为屏幕阅读器提供适当的 ARIA 标签

## Git 提交规范

```
feat: 添加用户登录功能
fix: 修复导航菜单在移动端的显示问题
style: 统一代码缩进格式
docs: 更新README安装说明
refactor: 重构API调用逻辑
perf: 优化图片加载性能
test: 添加表单验证测试
```

## 学习资源提示

当遇到复杂概念时,建议查阅:

- MDN Web Docs (权威的 Web 技术文档)
- Can I Use (浏览器兼容性查询)
- Web.dev (Google 的 Web 最佳实践)

## 特殊说明

- **问题优先**: 如果不确定某个实现,请先提出问题而不是直接给出可能错误的代码
- **解释优先**: 提供代码时,始终解释其工作原理
- **替代方案**: 如果有多种实现方式,请列出并说明各自的优缺点
- **向后兼容**: 注明代码的浏览器兼容性要求

---

**记住**: 好的代码不仅是能运行的代码,更是易于理解、维护和扩展的代码。