## 快速开始

### 环境要求
- **Python**: 3.12+
- **uv**: 快速的 Python 包管理器

安装 uv（macOS）：
```bash
brew install uv
```

### 安装依赖
```bash
uv venv
uv sync
```

### 配置环境变量
在项目根目录创建 `.env` 文件：
```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-4o-mini
```

### 运行分类
- 输入：默认读取 `test2.csv`（需要包含 `bmh, ocr, cleaned_ocr, cluster` 等列）
- 输出：`test2_classified.csv`

命令：
```bash
uv run python topic_classification.py
```

### 运行统计分析
- 默认输入：`test2_classified.csv`
- 默认输出：`classified_stats.csv`

使用默认参数：
```bash
uv run python analyze.py
```

自定义输入/输出路径：
```bash
uv run python analyze.py --input data/test2_classified.csv --output data/classified_stats.csv
```

### 说明
- 分析阶段会忽略“其他/其它/other/others”等一级标签。
- “关键词 (二级标签)”为该一级下所有二级标签（按频次降序，用“、”连接）。

