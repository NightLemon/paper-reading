# Paper Reading

计算机与软件系统工程方向的论文精读笔记与知识点沉淀。

📖 **在线阅读**：<https://nightlemon.github.io/paper-reading/>

每篇论文一个目录，包含原文（PDF + HTML）与一份六段式报告：元信息 → 摘要速览 → 细读 → 关键问题解析 → 可迁移知识点 → 批判与开放问题。跨论文复用的概念抽到 `docs/concepts/` 单独成篇，与论文笔记双向链接。

## 本地预览

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdocs serve
```

打开 <http://127.0.0.1:8000>，改动 Markdown 会自动热重载。

## 目录

```
docs/
├── index.md                  总索引
├── reading-list.md           待读队列
├── topics/<主题>/<年份>-<论文短名>/
│   ├── README.md             六段式报告
│   ├── paper.pdf
│   └── paper.html
└── concepts/                 跨论文知识点
templates/                   笔记模板（不发布到站点）
mkdocs.yml                   站点配置（新增论文时需同步 nav）
```

## 部署

推送到 GitHub 后，[.github/workflows/deploy.yml](.github/workflows/deploy.yml) 会在 `main` 分支每次 push 时自动构建并发布到 `gh-pages` 分支。首次使用需要在仓库 **Settings → Pages** 中把 Source 设为 `Deploy from a branch`，分支选 `gh-pages` / `(root)`。

## 多 agent 并行协作

部分工作以 GitHub Issue 形式拆分给其他 agent 并行完成。接单前请先读 [.github/copilot-instructions.md](.github/copilot-instructions.md) 的「多 agent 并行协作规则」一节：**每个 issue 只能修改它在「独占文件」一节列出的文件，共享索引文件（`docs/index.md`、`docs/reading-list.md`、`docs/concepts/README.md`、`docs/topics/*/README.md`、`mkdocs.yml`）一律不要碰。**
