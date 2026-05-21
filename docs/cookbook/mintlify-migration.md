# SGL-JAX Cookbook → Mintlify 迁移计划

> **文档性质**：长期跟踪文档。本次 PR 不一定完成全部步骤，后续相关 PR 都应回看本文并更新"当前进度"小节。
>
> **本文不被 cookbook 用户阅读**，是给 cookbook 维护者用的内部文档；同 `cookbook-recipe-design.md` / `2026-05-18-cookbook-research.md` 一样属于 author tooling。

## 1. 为什么要迁

当前 cookbook 是裸 markdown，在 GitHub 网页上读体验有限：

- 没有侧边导航树（只能靠 `index.md` 手动列表）
- 没有全文搜索
- 没有跨文件跳转的全局上下文（在哪个 vendor 哪个 recipe 上需要回到 url 栏看）
- 没有 Card / Tabs / Note 这些视觉组件，硬件配置矩阵、单/多机命令切换只能用裸 markdown 表格表达

目标：让 sglang-jax cookbook 跟 sglang 上游同等的阅读体验。

## 2. 调研结论（sglang 上游权威）

- **sglang 主仓**用 [Mintlify](https://mintlify.com/) 渲染文档，**不是** Docusaurus（sgl-cookbook 是历史代码库，已迁移到 `sglang/docs_new/`）
- 主站：[docs.sglang.io](https://docs.sglang.io/)
- 配置文件：`docs_new/docs.json`（`$schema: https://mintlify.com/docs.json`，`theme: aspen`）
- 文件后缀：`.mdx`（Markdown + JSX）
- vendor 目录命名：PascalCase（`Qwen/` `DeepSeek/` `Llama/` `GLM/` `InclusionAI/` `Moonshotai/` `Xiaomi/` `Google/` ...）
- 文件命名：`Model-Version.mdx`（`GLM-5.mdx`, `Ling-2.6.mdx`, `DeepSeek-R1.mdx`, `Llama3.1.mdx`）
- 视觉组件：`<Card>` `<CardGroup>` `<Tabs>` `<Steps>` `<CodeGroup>` `<Note>` `<Warning>` `<Tip>` `<Frame>` `<Accordion>`（Mintlify 预设组件库，import 即用）
- 自定义 React：sglang 团队还为重型 recipe 写了 React 交互组件（如 `<GLM5Deployment />`，按硬件/量化/特性点选生成命令），从 `/src/snippets/` import；这条 sglang-jax 暂不跟进

## 3. 阶段化计划

### Step 1 · 搭框架（一次性）

新增 3 个东西：

```
docs/cookbook/
├── docs.json       ← Mintlify 配置（name/theme/colors/logo/navigation 树）
├── index.mdx       ← cookbook 首页（intro）
└── (可选) logo/, favicon.png, fonts/
```

`docs.json` 的核心是 `navigation.tabs[].groups[].pages[]` 树，定义侧边栏。按 vendor 分组，列出每个 vendor 下的 recipe。

### Step 2 · 文件格式迁移

20 个 recipe + 顶层 index：
- `git mv X.md X.mdx`
- 顶部加 Mintlify frontmatter：
  ```yaml
  ---
  title: "Recipe Title"
  description: "One-line SEO description"
  ---
  ```
- 文件内容**不动**（裸 markdown 在 .mdx 里完全合法）

### Step 3 · 视觉组件渐进升级（可选，渐进做）

按价值优先级：

| 升级 | 优先级 | 替换什么 |
|---|---|---|
| `<CardGroup>` + `<Card>` | 高 | §1 Variants 列表 → 卡片网格（更易扫读） |
| `<Tabs>` + `<Tab>` | 高 | §2.3 单机 / 多机 launch 命令 → 标签切换（减少滚动） |
| `<Note>` / `<Warning>` / `<Tip>` | 中 | "⚠️ 提示" 段落 → 彩色高亮框 |
| `<CodeGroup>` | 中 | §3.1 curl + Python 客户端 → 多语言切换 |
| `<Steps>` + `<Step>` | 低 | §2.3 SkyPilot "Step 1 / Step 2" → 可视化步骤序列 |

每个 recipe 不必一次全升级，按需。

### Step 4 · 本地预览

```bash
npm i -g mintlify
cd docs/cookbook && mintlify dev
# 浏览器开 http://localhost:3000
```

### Step 5 · 部署（独立 PR，需 infra 决策）

候选方案：

| 方案 | 成本 | 说明 |
|---|---|---|
| Mintlify SaaS | 免费 plan 仅 1 项目 / $150+/月 起 | 连 GitHub 自动 build，零运维 |
| Vercel | 免费 | 需配 `mintlify build` 输出静态文件 |
| GitHub Pages | 免费 | CI 跑 `mintlify build` 推到 `gh-pages` |
| 自托管 | 服务器成本 | nginx serve mintlify 静态产物 |

依赖项：域名（`cookbook.sglang-jax.io`？还是嵌入 `sglang.io/cookbook/jax/...`？）、Mintlify 账号 / Vercel 账号、CI 配置。

## 4. 当前进度

- [ ] **Step 1** —— 待启动
- [ ] **Step 2** —— 待启动
- [ ] **Step 3** —— 待 Step 1+2 完成后渐进做
- [ ] **Step 4** —— 待 Step 1+2 完成
- [ ] **Step 5** —— 待 infra 决策

完成本节后请更新本文勾选状态。

## 5. 未决问题（待 owner 拍板）

| 问题 | 候选 | 待决断 |
|---|---|---|
| **logo / favicon** | (a) 项目方提供 sglang-jax logo；(b) 不设，纯文本名；(c) 临时用 sglang 上游 logo 占位 | 涉及视觉品牌 |
| **站点 URL** | (a) 独立 `cookbook.sglang-jax.io`；(b) 嵌入 `docs.sglang-jax.io/cookbook`；(c) sglang 主站子 tab `sglang.io/cookbook/jax/...` | 涉及站点边界 |
| **跟 sglang 主站合并 vs 独立站** | 合并：用户一处找全所有 cookbook；独立：sglang-jax 团队完全自主，不受上游 review 节奏限制 | 跨团队协调 |
| **部署 infra** | Mintlify SaaS / Vercel / GH Pages / 自托管 | Step 5 启动前必决 |
| **自定义 React 组件**（Step 3 之外） | 现阶段不做；未来需要时再单建 `/src/snippets/` | 跟价值评估 |

## 6. 跨 PR 协调约定

- 后续任何动 cookbook 文档的 PR 都先 grep 本文，看是否需要更新"当前进度"或"未决问题"
- Step 3 渐进升级可以**任何 recipe PR 顺手做**（不必单独立项），但应在 PR description 中标注"includes Mintlify component upgrade for §1 Variants"等
- Step 5 部署 PR 必须独立，且 PR description 链接到本文

## 7. 参考链接

- [Mintlify docs](https://mintlify.com/docs)
- [Mintlify 预设组件清单](https://mintlify.com/docs/components/accordions)
- [sglang docs_new (上游权威)](https://github.com/sgl-project/sglang/tree/main/docs_new)
- [sglang 部署的 docs.json 模板](https://github.com/sgl-project/sglang/blob/main/docs_new/docs.json)
- 本仓 [`cookbook-recipe-design.md`](cookbook-recipe-design.md) — recipe 5 段式 schema
- 本仓 [`2026-05-18-cookbook-research.md`](2026-05-18-cookbook-research.md) — cookbook 整体设计原则
