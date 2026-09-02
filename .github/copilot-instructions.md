# Copilot 协作约定

本仓库是计算机与软件系统工程方向的论文精读笔记库。在此仓库中工作时遵循以下约定。

## 语言与文风

- 笔记一律用**中文**书写，专业术语保留英文原词（如 prefix caching、continuous batching、p99），不强行翻译。
- 采用直接、肯定的表述：说清主体、动作与关系。避免「不是 X，而是 Y」这类在 X 从未被引入时的对比框架；只有当两个选项都在上下文中真实存在时才使用对比，并说明其关系。
- 表达证据边界时用「当前证据覆盖 A；B 需要单独验证」，而非引入无关的否定。
- 论文中的数字必须注明实验条件（模型、硬件、负载、基线），不得脱离条件单独引用。
- 不确定的内容标注 `<待确认>` 或写明推断依据，不编造论文中不存在的细节。

## 目录约定

本仓库用 **MkDocs Material** 构建文档站，所有笔记内容位于 `docs/`。

```
docs/
├── index.md
├── reading-list.md
├── topics/<主题>/<年份>-<论文短名>/
│   ├── README.md
│   ├── paper.pdf
│   └── paper.html
└── concepts/<概念>.md
templates/{paper-note,concept}.md   # 在仓库根目录，MkDocs 会排除 docs/templates/
mkdocs.yml
```

- 主题取值：`foundations` `llm-serving` `llm-training` `distributed-systems` `networking` `llm-applications`。
- 论文目录名与概念文件名均用 kebab-case，论文目录带发表年份前缀。
- 所有笔记必须带 YAML frontmatter，字段以模板为准。
- 页面间用相对路径链接到 `.md` 文件（MkDocs 会自动转成站点 URL）。中文标题锚点沿用 GitHub 风格 slug。

## 论文报告的六段式结构

1. 元信息 2. 摘要速览（5 分钟版） 3. 细读（按原文章节对齐） 4. 关键问题解析（Q&A） 5. 可迁移知识点（链接到 `concepts/`） 6. 批判与开放问题

细读部分的章节标题必须与原文实际章节一一对应，先读原文确认章节结构再写。

## 原文格式偏好

优先级：arXiv LaTeXML HTML > ar5iv HTML > PDF。写笔记前优先读 HTML 版本（章节、公式 `alttext`、图注结构完整），PDF 仅作补充。

## 知识点沉淀

- 跨论文复用的概念抽到 `docs/concepts/`，一个概念一篇，内容独立于任何单篇论文。
- 双向链接：论文笔记第 5 节链接概念页，概念页「出现在哪些论文里」反向链接论文笔记。

## 每次新增/更新笔记后必须同步

- `docs/index.md` 的论文索引表与主题分区篇数
- 对应 `docs/topics/<主题>/README.md` 的论文表
- `docs/concepts/README.md` 的概念索引表
- `docs/reading-list.md` 中该论文的勾选状态与「已进入精读」表
- **`mkdocs.yml` 的 `nav`**（新增页面必须手动登记，否则不会出现在侧边导航）

改完后用 `.\.venv\Scripts\python.exe -m mkdocs build --clean` 验证，确保没有链接告警。

## 多 agent 并行协作规则

本仓库会把部分工作拆成 GitHub Issue 交给其他 agent 并行完成。**接单时必须遵守文件归属约定**：

- 每个 issue 在「独占文件」一节列出它**唯一有权修改**的文件。只改这些文件。
- 以下**共享索引文件一律不要修改**（已由主线预先登记好占位）：
  `docs/index.md` · `docs/reading-list.md` · `docs/concepts/README.md` · `docs/topics/*/README.md` · `mkdocs.yml`
- 需要新增未预留的概念页时，创建文件即可，并在 PR 描述里列出应该补进索引的行，由主线整合。
- 每个工单开独立分支（建议 `paper/<短名>` 或 `chore/<主题>`），完成后提 PR，不直推 `main`。
- 提交前用 `mkdocs build --clean` 验证无链接告警。若告警来自尚未完成的其他工单，在 PR 里说明即可。

## 新论文发现流程

用户维护 `docs/reading-list.md` 待读队列。Copilot 定期补充候选并给出**推荐理由**（说明它与用户当前关注方向的关系），用户勾选 `[x]` 后再进入精读。未经勾选不直接写完整报告。
