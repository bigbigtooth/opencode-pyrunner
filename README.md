# OpenCode Python Runner

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.0-orange)](https://github.com/opencode/pyrunner)

一个 Python 封装库，用于通过 Python 代码驱动 opencode AI 编程助手。支持单任务目录创建、模型选择、上下文设置、多轮对话、运行监控和中断控制。

## 功能特性

- **独立任务目录** - 为每个任务创建隔离的工作目录
- **模型选择** - 支持指定使用的 AI 模型
- **上下文设置** - 支持设置系统提示/上下文
- **多轮对话** - 通过会话 ID 支持连续对话
- **实时监控** - 监控任务运行状态和输出
- **中断控制** - 支持优雅中断和强制终止任务
- **自动保存** - 自动保存对话为 Markdown 文件
- **模型源配置** - 支持为不同场景配置不同模型

## 安装

### 从源码安装（推荐开发使用）

```bash
git clone https://github.com/opencode/pyrunner.git
cd pyrunner
pip install -e .
```

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 环境要求

- **Python 3.12+** - 需要支持类型提示
- **opencode CLI** - 必须已安装并配置好

确认 opencode 已安装：

```bash
opencode --version
```

## 快速开始

### 基本用法

```python
from opencode_pyrunner import OpenCodeRunner

# 创建运行器
runner = OpenCodeRunner(task_name='hello')

# 执行任务
result = runner.run('Say hello in Chinese')

# 打印结果
print(result['output'])
```

### 指定模型

```python
from opencode_pyrunner import OpenCodeRunner

# 查看可用模型: opencode models
runner = OpenCodeRunner(
    task_name='my_task',
    model='opencode/claude-sonnet-4-5'
)

result = runner.run('What is Python?')
print(result['output'])
```

### 多轮对话

```python
from opencode_pyrunner import OpenCodeRunner

runner = OpenCodeRunner(task_name='conversation')

# 第一轮
result1 = runner.run('What is 2+2?')
print(result1['output'])  # '4'

# 第二轮 - 自动使用同一会话
result2 = runner.run('Now multiply that by 3')
print(result2['output'])  # '12'

# 查看会话 ID
print(runner.get_session_id())
```

## 模型源配置

支持为不同场景配置不同的模型：

### 代码方式配置

```python
from opencode_pyrunner import OpenCodeRunner, ModelSource, ModelSourceType

# 创建模型源配置
model_source = ModelSource()

# 配置代码任务模型
model_source.set_model(
    ModelSourceType.CODE,
    baseurl="https://api.openai.com",
    apikey="sk-your-api-key",
    model="gpt-4",
    temperature=0.3
)

# 配置写作任务模型
model_source.set_model(
    ModelSourceType.WRITE,
    baseurl="https://api.openai.com",
    apikey="sk-your-api-key",
    model="gpt-3.5-turbo",
    temperature=0.8
)

# 创建运行器
runner = OpenCodeRunner(
    task_name='multi_task',
    model_source=model_source
)

# 执行编码任务
result1 = runner.run(
    "Write a Python function to calculate fibonacci",
    model_source_type=ModelSourceType.CODE
)

# 执行写作任务
result2 = runner.run(
    "Write a short poem about coding",
    model_source_type=ModelSourceType.WRITE
)
```

### 从 .env 文件加载配置

创建 `.env` 文件：

```env
# Code 任务模型
MODEL_CODE_BASEURL=https://api.openai.com/v1
MODEL_CODE_APIKEY=sk-your-api-key
MODEL_CODE_MODEL=gpt-4
MODEL_CODE_TEMPERATURE=0.3
MODEL_CODE_MAX_TOKENS=4096

# Write 任务模型
MODEL_WRITE_BASEURL=https://api.openai.com/v1
MODEL_WRITE_APIKEY=sk-your-api-key
MODEL_WRITE_MODEL=gpt-3.5-turbo
MODEL_WRITE_TEMPERATURE=0.8
MODEL_WRITE_MAX_TOKENS=2048
```

加载配置：

```python
from opencode_pyrunner import OpenCodeRunner, ModelSource

# 从 .env 文件加载
model_source = ModelSource.from_env_file('.env')
runner = OpenCodeRunner(task_name='env_task', model_source=model_source)

# 执行任务
result = runner.run("分析这段代码", model_source_type=ModelSourceType.CODE)
```

更多配置示例请参考 `.env.example` 文件。

## 监控与中断

### 异步启动并监控

```python
import time
from opencode_pyrunner import OpenCodeRunner

runner = OpenCodeRunner(task_name='monitor')

# 异步启动，不等待完成
result = runner.run('Write a long story', wait=False)

print(f"State: {result['state']}")  # 'running'

# 循环监控
while runner.is_running():
    time.sleep(1)
    state = runner.get_state()
    output = runner.get_current_output()
    print(f"State: {state.value}, Output length: {len(output)}")

# 等待完成
final_result = runner.wait()
print(final_result['output'])
```

### 中断任务

```python
import time
from opencode_pyrunner import OpenCodeRunner

runner = OpenCodeRunner(task_name='interrupt_test')
runner.run('List all files recursively', wait=False)

# 运行一段时间后中断
time.sleep(3)

if runner.interrupt():
    print("已发送中断信号")
else:
    print("任务已完成，无法中断")

# 查看最终状态
print(runner.get_state())  # RunState.INTERRUPTED
```

### 强制终止

```python
# 如果优雅中断无效，可以强制终止
runner = OpenCodeRunner(task_name='force_kill')
runner.run('任务', wait=False)

time.sleep(2)

# 尝试优雅中断
if not runner.interrupt():
    # 如果不行，强制终止
    runner.kill()
```

## 保存结果

```python
from opencode_pyrunner import OpenCodeRunner

runner = OpenCodeRunner(task_name='save_test')
runner.run('问题1')
runner.run('问题2')

# 保存到任务目录的 conversation.md
filepath = runner.save_markdown()
print(f"已保存到: {filepath}")

# 保存到自定义路径
filepath = runner.save_markdown('/path/to/output.md')

# 获取 Markdown 字符串而不保存
markdown_text = runner.get_conversation_markdown()
print(markdown_text)
```

## API 参考

### OpenCodeRunner

主运行器类，用于执行 opencode 任务。

#### 初始化参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `task_name` | str | 是 | 任务名称，用于创建任务目录 |
| `model` | str | 否 | 使用的模型标识符 |
| `context` | str | 否 | 系统提示/上下文 |
| `workspace` | str | 否 | 工作空间根目录，默认为当前目录 |
| `model_source` | ModelSource | 否 | 模型源配置 |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `run(message, ...)` | 执行 opencode 命令 |
| `wait(timeout=None)` | 等待任务完成 |
| `interrupt()` | 优雅中断任务 |
| `kill()` | 强制终止任务 |
| `is_running()` | 检查是否正在运行 |
| `get_state()` | 获取当前状态 |
| `get_current_output()` | 获取当前输出 |
| `get_session_id()` | 获取会话 ID |
| `save_markdown(path=None)` | 保存对话为 Markdown |
| `get_conversation_markdown()` | 获取 Markdown 字符串 |

### ModelSource

模型源配置类，用于管理不同场景的模型配置。

#### 方法

| 方法 | 说明 |
|------|------|
| `set_model(source_type, ...)` | 设置指定场景的模型 |
| `get_model(source_type)` | 获取指定场景的模型配置 |
| `has_model(source_type)` | 检查是否配置了指定场景 |
| `from_env_file(filepath)` | 从 .env 文件加载配置 |
| `from_dict(data)` | 从字典创建 |

### ModelSourceType

模型源类型枚举：

- `PLAN` - 执行规划任务
- `THINK` - 执行思考任务
- `CODE` - 执行编码任务
- `WRITE` - 执行写作任务
- `DEFAULT` - 执行普通任务

### RunState

运行状态枚举：

- `IDLE` - 初始状态
- `RUNNING` - 运行中
- `COMPLETED` - 已完成
- `INTERRUPTED` - 已中断
- `ERROR` - 出错

## 完整示例

```python
import time
from opencode_pyrunner import OpenCodeRunner, ModelSource, ModelSourceType, RunState

# 创建模型源配置
model_source = ModelSource()
model_source.set_model(
    ModelSourceType.CODE,
    baseurl="https://api.openai.com",
    apikey="sk-your-api-key",
    model="gpt-4",
    temperature=0.3
)

# 创建运行器
runner = OpenCodeRunner(
    task_name='example',
    model_source=model_source,
    context='You are a Python expert. Answer in Chinese.'
)

# 执行任务
result = runner.run(
    'Write a Python function to calculate factorial',
    model_source_type=ModelSourceType.CODE,
    wait=True
)

print(f"输出: {result['output']}")
print(f"会话 ID: {result['session_id']}")
print(f"任务目录: {result['task_dir']}")
print(f"状态: {result['state']}")

# 继续对话
result2 = runner.run('Add error handling to the function')
print(f"第二轮输出: {result2['output']}")

# 保存对话
filepath = runner.save_markdown()
print(f"对话已保存到: {filepath}")
```

## 命令行用法

安装后可以直接使用命令行：

```bash
# 基本用法
python -m opencode_pyrunner "What is Python?"

# 指定任务名称
python -m opencode_pyrunner "What is Python?" --task-name my_task

# 指定模型
python -m opencode_pyrunner "What is Python?" --model opencode/claude-sonnet-4-5

# 指定上下文
python -m opencode_pyrunner "What is Python?" --context "Answer in Chinese"

# 保存对话
python -m opencode_pyrunner "What is Python?" --save-markdown

# 查看帮助
python -m opencode_pyrunner --help
```

## 开发

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black opencode_pyrunner.py
```

### 类型检查

```bash
mypy opencode_pyrunner.py
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 文档

详细文档请参考 [doc/opencode_runner_docs.html](doc/opencode_runner_docs.html)
