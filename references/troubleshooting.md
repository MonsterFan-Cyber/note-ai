# Troubleshooting — 常见问题排查

> 当笔记产出过程中出现以下问题，按对应条目处置。本文件仅在遇到具体问题时才需读取全文。

## 1. Mermaid 绘制失败或语法报错

**现象**：图渲染不出来、节点不连线、`classDef` 报错、引号字符（如 `ᵀ`、`⊙`）导致解析失败。

**处置**：

1. **保持语法最小化**：节点 label 内避免特殊字符裸用——`ᵀ` 用 `^T` 或写 `转置`；符号 `⊙` 用 `mul`；引号冲突时节点写 `["..."]` 嵌套转义。
2. **分步验证**：先画只有节点+连线的骨架图确认渲染，再逐批加入 `classDef` 与 `:::class` 引用。
3. **长 label 拆分**：三行节点用 `\n` 分隔，避免单个 label 超过渲染器宽度。
4. **先检查环境**：验证工具链是否具备离屏渲染条件。已踩过的坑——
   - `mmdc`（mermaid-cli）依赖 puppeteer/Chromium；若环境未下载浏览器（如设置了 `PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true`），`mmdc` 会**静默输出 0 字节 SVG 且不报错**，勿误判为渲染成功；
   - 在 node 无浏览器环境直接 `require('mermaid')` 调 `parse/render`，会因缺 DOM 报 `DOMPurify.sanitize is not a function` 等环境错误——这是环境问题而非语法问题，**不要反复折腾 jsdom/DOMPurify 注入**，浪费时间。
5. **退回只检查语法**：若渲染工具链不可用，**放弃离屏渲染验证**，改为人工语法检查：核对 `flowchart` 声明与节点/边/`classDef` 结构，确认特殊字符（`|`、`[`、`→`、`<br/>`、组合音调符号 `P̂`/`ξ̂` 等）均在 `"..."` 引号内、无裸 `subgraph` 嵌套。检查结果输出一般用 ```txt``` 包裹。
6. **语法自检**：绘制完成后用第 7 节检查清单中的"Mermaid 语法已检查"项核对。

## 2. PyMuPDF 未安装

**现象**：`import fitz` 报 ModuleNotFoundError。

**处置**：`pip install pymupdf` 后重试。若 pip 不可用（内网/无权限），改用备选方案：`pdfplumber` 或 `pdf2image`（Poppler 转 PNG 后按区域裁剪）。裁剪质量要求不变（见 SKILL.md 4.2）。

## 3. 双栏 PDF 图区域裁剪不准

**现象**：裁剪结果混入相邻栏正文，或图注被裁进图片。

**处置**：

1. 先按图注（`Figure N.`）所在栏确定裁剪列：比较 caption 的 x0 与页面中线，左栏或右栏。
2. 收集 `get_drawings()` 的路径 bbox 时，**只保留 x 范围与图注同栏**的路径，再聚类。
3. 若矢量路径稀疏（该图实际是位图），改用 `get_images()` 定位图像 bbox 直接裁剪，忽略矢量聚类结果。
4. 裁剪后必须核对：图区域下沿 > 图注上沿（不得裁进 caption）；上沿不含页眉/正文。不满足则调整 rect 重试。

## 4. 图片 HTML 标签渲染失效（个别渲染器）

**现象**：`<img>` 标签在部分渲染器（如某些旧版 VitePress）被过滤或宽度控制不生效。

**处置**：优先保留 HTML 写法（GitHub/Obsidian/Typora 均支持）。若目标平台确实不支持，退化为 `![alt](images/xxx.png)` 并在图下说明中额外写一句"图片为原宽"，确保读者明白布局原因。**不得**为了兼容放弃限宽要求本身，必要时缩小源图尺寸。

## 5. 大纲临时文件遗漏未删

**现象**：`.note/` 下残留 `_` 前缀大纲文件，违反 SKILL.md「撰写工作流」第 5 步的收尾硬步骤。

**处置**：填充完成后主动列出 `.note/` 下所有 `_` 前缀文件并删除；若用户已修改过大纲且需要留档，在交付前询问用户是否删除。

## 6. 版本号 / 文档不同步

**现象**：`SKILL.md` 的 frontmatter `version` 与 `CHANGELOG.md` 最新条目不一致。

**处置**：每次修改 SKILL.md、模板或新增资源后，递增 `version` 并在 `CHANGELOG.md` 顶部追加条目（日期格式 `YYYY-MM-DD`），随后同步检查 `README.md` 目录结构与版本描述。

> **版本号有三处，必须同时改**：① `SKILL.md` frontmatter 的 `version`；② `README.md` 顶部 shields.io 徽章 `version-x.y.z`（最容易漏，已实际漏过一次）；③ `README.md` 末尾「📅 版本」段的"当前版本 **x.y.z**"。
> 核对命令：`Select-String -Path SKILL.md,README.md -Pattern 'version[: ]|version-|当前版本'`，三处应完全一致。