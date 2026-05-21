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
- 文件后缀：`.md`（Markdown + JSX）
- vendor 目录命名：PascalCase（`Qwen/` `DeepSeek/` `Llama/` `GLM/` `InclusionAI/` `Moonshotai/` `Xiaomi/` `Google/` ...）
- 文件命名：`Model-Version.md`（`GLM-5.md`, `Ling-2.6.md`, `DeepSeek-R1.md`, `Llama3.1.md`）
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
- `git mv X.md X.md`
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

#### 上游调研结论（2026-05-21）

sglang 主仓的 docs 是**双轨**：
- 老 Sphinx docs（`docs/`）→ `.github/workflows/release-docs.yml` 触发 → 推到 `sgl-project.github.io`
- 新 Mintlify docs (`docs_new/`) → **仓里没有任何 deploy workflow** → 即 sglang 主仓的 Mintlify 是用 **Mintlify SaaS**：在 mintlify.com 控制台连 GitHub repo，平台自动 build & deploy 到 `docs.sglang.io`，仓代码完全干净。

这是 sglang-jax 跟随上游最直接的方式。

#### 三条候选路径

| 路径 | 工作流 | 公共 URL | 成本 | 上手 |
|---|---|---|---|---|
| **A. Mintlify SaaS（跟上游一致）** | mintlify.com 注册 → 连 `sgl-project/sglang-jax` repo → 指定 docs root = `docs/cookbook/` → 平台自动 build & deploy；每次 main push 触发 | 自动得到 `<project>.mintlify.app`；CNAME 可绑自定义域 | Free plan：1 项目、无限页面、`*.mintlify.app` 子域名。自定义域名 + 移除 Mintlify 品牌：Pro $150/mo | 10 min（仅 console 配置） |
| **B. GitHub Pages（自部署）** | 加 `.github/workflows/cookbook-deploy.yml`：在 main push 时 `cd docs/cookbook && mint build` → 推 `dist/` 到 `gh-pages` 分支 → GitHub Pages 服务静态文件 | `sgl-project.github.io/sglang-jax/cookbook/`，CNAME 可绑自定义域 | 全免费 | ~半天（写 + 调 CI；首次 build 缓存 npm/mint deps） |
| **C. Vercel / Cloudflare Pages** | 在 Vercel/CF 控制台连 GitHub repo → build command `cd docs/cookbook && mint build` → output dir `docs/cookbook/dist/` | `<project>.vercel.app` 或 `<project>.pages.dev`，CNAME 可绑自定义域 | 全免费（合理流量内） | ~30 min（console 配置 + 微调 build） |

#### 推荐

- **A 最贴近"跟 sglang 体验一致"**：连仓即用，无 CI 代码污染，未来 sglang 的 docs.json / 主题升级我们能直接抄。短板是需要 Mintlify 账号 + 自定义域名要付费。
- **B 是回退方案**：项目方不想引入外部 SaaS / 不愿付费时用 GitHub Pages，完全自给自足；代价是写并维护 deploy workflow。
- C 适中：免费 + 控制台配置，但又引入了第二个 SaaS（Vercel/CF），不如 A 顺、不如 B 净。

#### 必决项（决策权不在文档作者，需 sglang-jax owner 拍）

| 项 | 候选 |
|---|---|
| **谁 own SaaS 账号 / 部署 infra** | sgl-project 组织级 / 个人 maintainer / sglang-jax 子团队 |
| **域名归属** | 不要自定义（用 `<auto>.mintlify.app`）/ 申请 `cookbook.sglang-jax.io` / 嵌入未来的 `docs.sglang-jax.io/cookbook` / 合并到 sglang 主站 `/cookbook/jax/...` |
| **PR scope** | 部署 PR 是否包含在当前 cookbook PR（不推荐 —— 现 PR 已经够大）/ 单开 PR |
| **私有 vs 公共** | 公共 URL 立即对外 / 先内部 review 内容稳定再公共部署 |
| **跟 sglang 主站合并 vs 独立站** | 独立 sglang-jax 站点（控制权完全自有）/ 申请并入 sglang 主站作为 JAX 子 tab（用户一处找全所有 cookbook） |

## 4. 当前进度

- [x] **Step 1** —— 完成（`docs.json` 已写入，sidebar 按 sglang 上游单顶级 group + 嵌套 vendor group 结构）
- [x] **Step 2** —— 完成（30 个文件加 frontmatter；后缀最终选 `.md` 而非 `.mdx`，原因：GitHub 识别 `.md` 的 YAML frontmatter 并自动隐藏，`.mdx` 不识别。Mintlify 同时支持，无功能损失。需要 JSX 组件的具体 page 可单独改 `.mdx`）
- [ ] **Step 3** —— 待启动；任何 recipe PR 顺手做即可
- [x] **Step 4** —— 完成（本地预览验证：`PATH="/opt/homebrew/opt/node@22/bin:$PATH" npx --yes mintlify@latest dev`，因为 mintlify CLI 不支持 node 25+，需要 LTS）
- [ ] **Step 5** —— 调研完成，方案已落 §3.5；待 owner 拍板 infra 后启动独立 PR

完成本节后请更新本文勾选状态。

## 5. 未决问题（待 owner 拍板）

| 问题 | 候选 | 待决断 |
|---|---|---|
| **logo / favicon** | (a) 项目方提供 sglang-jax logo；(b) 不设，纯文本名；(c) 临时用 sglang 上游 logo 占位 | 涉及视觉品牌 |
| **自定义 React 组件**（Step 3 之外） | 现阶段不做；未来需要时再单建 `/src/snippets/` | 价值评估 |

**部署相关决策** 已统一在 §3 Step 5 内列详细方案 + 必决项（路径 A/B/C、域名归属、PR scope、私公开、跟 sglang 主站合并 vs 独立），不在此重复。

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
