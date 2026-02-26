"""
OpenCode Python Runner
=======================

一个 Python 封装库，用于通过 Python 代码驱动 opencode AI 编程助手。
支持单任务目录创建、模型选择、上下文设置、多轮对话、运行监控和中断控制。

作者: OpenCode User
版本: 1.1.0
"""

import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """
    模型配置类
    
    Attributes:
        baseurl: API 基础 URL
        apikey: API 密钥
        model: 模型名称
        temperature: 温度参数 (0-2)
        max_tokens: 最大 token 数
    """
    
    def __init__(
        self,
        baseurl: str = "",
        apikey: str = "",
        model: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.baseurl = baseurl
        self.apikey = apikey
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "baseurl": self.baseurl,
            "apikey": self.apikey,
            "model": self.model,
        }
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            baseurl=data.get("baseurl", ""),
            apikey=data.get("apikey", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
        )


class ModelSourceType(Enum):
    """
    模型源类型枚举
    """
    PLAN = "plan"       # 执行plan任务
    THINK = "think"     # 执行思考任务
    CODE = "code"       # 执行编码任务
    WRITE = "write"     # 执行写作任务
    DEFAULT = "default" # 执行普通任务


def load_env_file(filepath: str) -> Dict[str, str]:
    """
    加载 .env 格式的配置文件
    
    Args:
        filepath: 配置文件路径
        
    Returns:
        键值对字典
    """
    env_vars = {}
    if not os.path.exists(filepath):
        logger.warning(f"配置文件不存在: {filepath}")
        return env_vars
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
    
    logger.info(f"已加载配置文件: {filepath}, 共 {len(env_vars)} 个配置项")
    return env_vars


def parse_env_model_config(env_vars: Dict[str, str]) -> "ModelSource":
    """
    从环境变量字典解析模型配置
    
    支持的格式:
        MODEL_PLAN_BASEURL=https://api.openai.com
        MODEL_PLAN_APIKEY=sk-xxx
        MODEL_PLAN_MODEL=gpt-4
        MODEL_PLAN_TEMPERATURE=0.7
        MODEL_PLAN_TOP_K=40
        MODEL_PLAN_MAX_TOKENS=4096
        
        MODEL_THINK_*
        MODEL_CODE_*
        MODEL_WRITE_*
        MODEL_DEFAULT_*
    
    Args:
        env_vars: 环境变量字典
        
    Returns:
        ModelSource 实例
    """
    sources: Dict[ModelSourceType, ModelConfig] = {}
    type_names = {
        "plan": ModelSourceType.PLAN,
        "think": ModelSourceType.THINK,
        "code": ModelSourceType.CODE,
        "write": ModelSourceType.WRITE,
        "default": ModelSourceType.DEFAULT,
    }
    
    for type_name, source_type in type_names.items():
        prefix = f"MODEL_{type_name.upper()}"
        
        baseurl = env_vars.get(f"{prefix}_BASEURL", "")
        apikey = env_vars.get(f"{prefix}_APIKEY", "")
        model = env_vars.get(f"{prefix}_MODEL", "")
        
        if not model:
            continue
        
        temp_str = env_vars.get(f"{prefix}_TEMPERATURE", "")
        max_tokens_str = env_vars.get(f"{prefix}_MAX_TOKENS", "")
        
        temperature = float(temp_str) if temp_str else None
        max_tokens = int(max_tokens_str) if max_tokens_str else None
        
        sources[source_type] = ModelConfig(
            baseurl=baseurl,
            apikey=apikey,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(f"解析模型配置 {type_name}: {model}")
    
    return ModelSource(sources=sources)


class ModelSource:
    """
    模型源配置类
    
    用于配置不同场景使用的模型，支持通过配置文件或代码方式设置。
    
    Attributes:
        sources: 存储各场景模型配置的字典
    """
    
    def __init__(self, sources: Optional[Dict[ModelSourceType, ModelConfig]] = None):
        self.sources = sources or {}
    
    def set_model(
        self,
        source_type: ModelSourceType,
        baseurl: str,
        apikey: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        设置指定场景的模型配置
        
        Args:
            source_type: 场景类型 (plan/think/code/write/default)
            baseurl: API 基础 URL
            apikey: API 密钥
            model: 模型名称
            temperature: 温度参数 (0-2)
            max_tokens: 最大 token 数
        """
        self.sources[source_type] = ModelConfig(
            baseurl=baseurl,
            apikey=apikey,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(f"设置模型源 {source_type.value}: {model}")
    
    def get_model(self, source_type: ModelSourceType) -> Optional[ModelConfig]:
        """
        获取指定场景的模型配置
        
        Args:
            source_type: 场景类型
            
        Returns:
            ModelConfig 或 None
        """
        return self.sources.get(source_type)
    
    def has_model(self, source_type: ModelSourceType) -> bool:
        """
        检查是否配置了指定场景的模型
        
        Args:
            source_type: 场景类型
            
        Returns:
            是否已配置
        """
        return source_type in self.sources
    
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, str]]) -> "ModelSource":
        """
        从字典创建 ModelSource
        
        Args:
            data: 字典格式的模型配置
            
        Returns:
            ModelSource 实例
        """
        sources = {}
        type_mapping = {
            "plan": ModelSourceType.PLAN,
            "think": ModelSourceType.THINK,
            "code": ModelSourceType.CODE,
            "write": ModelSourceType.WRITE,
            "default": ModelSourceType.DEFAULT,
        }
        
        for key, value in data.items():
            if key in type_mapping:
                sources[type_mapping[key]] = ModelConfig.from_dict(value)
        
        return cls(sources=sources)
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """
        转换为字典格式
        
        Returns:
            字典格式的模型配置
        """
        result = {}
        for source_type, config in self.sources.items():
            result[source_type.value] = config.to_dict()
        return result
    
    @classmethod
    def from_env_file(cls, filepath: str = ".env") -> "ModelSource":
        """
        从 .env 文件加载模型配置
        
        支持的格式:
            MODEL_PLAN_BASEURL=https://api.openai.com
            MODEL_PLAN_APIKEY=sk-xxx
            MODEL_PLAN_MODEL=gpt-4
            MODEL_PLAN_TEMPERATURE=0.7
            MODEL_PLAN_TOP_K=40
            MODEL_PLAN_MAX_TOKENS=4096
            
            MODEL_CODE_BASEURL=...
            MODEL_CODE_APIKEY=...
            MODEL_CODE_MODEL=...
        
        Args:
            filepath: .env 文件路径，默认当前目录的 .env
        
        Returns:
            ModelSource 实例
        """
        env_vars = load_env_file(filepath)
        return parse_env_model_config(env_vars)


class RunState(Enum):
    """
    运行状态枚举
    
    Attributes:
        IDLE: 初始状态，未开始运行
        RUNNING: 正在运行中
        COMPLETED: 已正常完成
        INTERRUPTED: 被用户中断
        ERROR: 运行出错
    """
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class OpenCodeRunner:
    """
    OpenCode 运行器类
    
    提供对 opencode CLI 的 Python 封装，支持创建独立任务目录、
    指定模型和上下文、多轮对话、实时监控和中断控制。
    
    Attributes:
        task_name: 任务名称，用于创建任务目录
        model: 使用的模型标识符
        context: 初始上下文/系统提示
        workspace: 工作空间根目录
        task_dir: 当前任务的工作目录
        session_id: 当前会话 ID，用于多轮对话
    
    Example:
        >>> runner = OpenCodeRunner(task_name='my_task')
        >>> result = runner.run('What is 2+2?')
        >>> print(result['output'])  # '4'
    """
    
    def __init__(
        self,
        task_name: str,
        model: Optional[str] = None,
        context: Optional[str] = None,
        workspace: Optional[str] = None,
        model_source: Optional[ModelSource] = None,
    ):
        """
        初始化 OpenCode 运行器
        
        Args:
            task_name: 任务名称，用于创建任务目录（必需）
            model: 使用的模型标识符，如 'opencode/claude-sonnet-4-5'
                   可用模型通过 `opencode models` 查看
            context: 初始上下文/系统提示，会作为文件附加到每次请求
            workspace: 工作空间根目录，默认为当前目录
            model_source: 模型源配置，用于不同场景使用不同模型
        
        Example:
            >>> # 基础用法
            >>> runner = OpenCodeRunner(task_name='test')
            
            >>> # 指定模型
            >>> runner = OpenCodeRunner(task_name='test', model='opencode/claude-sonnet-4-5')
            
            >>> # 指定模型源
            >>> model_source = ModelSource()
            >>> model_source.set_model(ModelSourceType.CODE, "https://api.openai.com", "sk-xxx", "gpt-4")
            >>> model_source.set_model(ModelSourceType.WRITE, "https://api.openai.com", "sk-xxx", "gpt-3.5-turbo")
            >>> runner = OpenCodeRunner(task_name='test', model_source=model_source)
            
            >>> # 从字典加载模型源
            >>> config = {
            ...     "code": {"baseurl": "https://api.openai.com", "apikey": "sk-xxx", "model": "gpt-4"},
            ...     "write": {"baseurl": "https://api.openai.com", "apikey": "sk-xxx", "model": "gpt-3.5-turbo"}
            ... }
            >>> model_source = ModelSource.from_dict(config)
            >>> runner = OpenCodeRunner(task_name='test', model_source=model_source)
        """
        self.task_name = task_name
        self.model = model
        self.context = context
        self.workspace = workspace or os.getcwd()
        self.model_source = model_source or ModelSource()
        
        # 创建唯一的任务目录
        self.task_dir = os.path.join(
            self.workspace, 
            f"task_{task_name}_{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(self.task_dir, exist_ok=True)
        logger.info(f"创建任务目录: {self.task_dir}")
        
        self.session_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.full_json_events: List[Dict[str, Any]] = []
        
        # 进程控制
        self._process: Optional[subprocess.Popen] = None
        self._state = RunState.IDLE
        self._output_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        
        logger.info(f"OpenCodeRunner 初始化完成: task_name={task_name}, model={model}")
    
    def run(
        self, 
        message: str, 
        context: Optional[str] = None, 
        wait: bool = True,
        model: Optional[str] = None,
        model_source_type: Optional[ModelSourceType] = None,
    ) -> Dict[str, Any]:
        """
        执行 opencode 命令
        
        Args:
            message: 发送给 opencode 的消息（必需）
            context: 此次请求的额外上下文，会覆盖默认 context
            wait: 是否等待完成，True 为阻塞等待，False 为异步启动
            model: 此次请求使用的模型，会覆盖默认 model 和 model_source 配置
            model_source_type: 指定使用哪个场景的模型 (plan/think/code/write/default)
        
        Returns:
            包含运行结果的字典:
            - output: 生成的输出文本
            - session_id: 会话 ID
            - task_dir: 任务目录
            - state: 当前状态 (wait=True 时)
            - json_events: JSON 事件列表 (wait=True 时)
        
        Example:
            >>> # 阻塞等待模式（默认）
            >>> runner = OpenCodeRunner(task_name='test')
            >>> result = runner.run('What is 2+2?')
            >>> print(result['output'])  # '4'
            
            >>> # 异步模式，不等待完成
            >>> result = runner.run('复杂任务...', wait=False)
            >>> print(result['state'])  # 'running'
            
            >>> # 使用模型源场景
            >>> runner = OpenCodeRunner(task_name='test', model_source=model_source)
            >>> result = runner.run('写一段代码', model_source_type=ModelSourceType.CODE)
            
            >>> # 临时指定模型
            >>> result = runner.run('任务', model='opencode/claude-sonnet-4-5')
        """
        logger.info(f"执行任务: {message[:50]}...")
        
        cmd = ["opencode", "run", "--format", "json"]
        
        # 确定使用的模型
        model_to_use = model  # 优先使用传入的 model 参数
        
        # 如果没有直接指定 model，则尝试从 model_source 获取
        if not model_to_use and model_source_type:
            model_config = self.model_source.get_model(model_source_type)
            if model_config:
                model_to_use = model_config.model
                logger.debug(f"使用模型源 {model_source_type.value}: {model_to_use}")
        
        # 如果仍然没有 model，则使用默认的 self.model
        if not model_to_use and self.model:
            model_to_use = self.model
        
        # 添加模型参数
        if model_to_use:
            cmd.extend(["--model", model_to_use])
            logger.debug(f"使用模型: {model_to_use}")
        
        # 添加会话参数（多轮对话）
        if self.session_id:
            cmd.extend(["--session", self.session_id])
            logger.debug(f"继续会话: {self.session_id}")
        
        working_dir = self.task_dir
        
        # 处理上下文
        context_to_use = context or self.context
        if context_to_use:
            context_file = os.path.join(self.task_dir, "context.md")
            with open(context_file, "w", encoding="utf-8") as f:
                f.write(context_to_use)
            cmd.extend(["--file", context_file])
            logger.debug(f"附加上下文文件: {context_file}")
        
        # 添加消息（使用 -- 分隔符避免参数解析问题）
        cmd.append("--")
        cmd.append(message)
        
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        # 启动子进程
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
        )
        
        self._state = RunState.RUNNING
        self._output_queue = queue.Queue()
        
        # 启动输出读取线程
        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()
        
        # 记录用户消息
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "context": context_to_use,
        })
        
        if wait:
            return self.wait()
        
        logger.info(f"任务已启动，状态: {self._state.value}")
        return {
            "session_id": self.session_id,
            "task_dir": self.task_dir,
            "state": self._state.value,
        }
    
    def _select_model(
        self,
        model: Optional[str] = None,
        model_source_type: Optional[ModelSourceType] = None,
    ) -> Optional[str]:
        """
        选择要使用的模型
        
        优先级: 传入的 model 参数 > model_source 配置 > 默认 self.model
        
        Args:
            model: 直接指定的模型
            model_source_type: 模型源类型
            
        Returns:
            选中的模型名称或 None
        """
        # 优先使用传入的 model 参数
        if model:
            return model
        
        # 其次尝试从 model_source 获取
        if model_source_type:
            model_config = self.model_source.get_model(model_source_type)
            if model_config:
                return model_config.model
        
        # 最后使用默认的 self.model
        return self.model
    
    def _build_command(
        self,
        message: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        model_source_type: Optional[ModelSourceType] = None,
    ) -> List[str]:
        """
        构建 opencode 命令
        
        Args:
            message: 发送给 opencode 的消息
            context: 此次请求的额外上下文
            model: 此次请求使用的模型
            model_source_type: 指定使用哪个场景的模型
            
        Returns:
            命令参数列表
        """
        cmd = ["opencode", "run", "--format", "json"]
        
        # 确定使用的模型
        model_to_use = self._select_model(model, model_source_type)
        
        if model_to_use:
            cmd.extend(["--model", model_to_use])
        
        # 添加会话参数（多轮对话）
        if self.session_id:
            cmd.extend(["--session", self.session_id])
        
        # 处理上下文
        context_to_use = context or self.context
        if context_to_use:
            context_file = os.path.join(self.task_dir, "context.md")
            cmd.extend(["--file", context_file])
        
        # 添加消息
        cmd.append("--")
        cmd.append(message)
        
        return cmd
    
    def _read_output(self):
        """
        内部方法：在后台线程中读取子进程输出
        
        将输出解析为 JSON 事件并放入队列，
        同时更新会话 ID 和运行状态。
        """
        if not self._process or not self._process.stdout:
            logger.warning("子进程 stdout 不可用")
            return
        
        try:
            for line in self._process.stdout:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        self.full_json_events.append(event)
                        self._output_queue.put(event)
                        
                        # 处理错误事件
                        if event.get("type") == "error":
                            logger.error(f"收到错误事件: {event.get('error')}")
                            self._state = RunState.ERROR
                        
                        # 处理文本事件
                        elif event.get("type") == "text":
                            text = event.get("part", {}).get("text", "")
                            self._output_queue.put(("text", text))
                        
                        # 处理步骤开始/结束事件
                        elif event.get("type") in ["step_start", "step_finish"]:
                            session_id = event.get("sessionID")
                            if session_id:
                                self._session_id = session_id
                                logger.debug(f"会话 ID 更新: {session_id}")
                            
                            if event.get("type") == "step_finish":
                                self._state = RunState.COMPLETED
                                logger.info("任务完成")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}")
        except Exception as e:
            logger.error(f"读取输出时出错: {e}")
        finally:
            if self._process:
                self._process.wait()
                if self._state == RunState.RUNNING:
                    self._state = RunState.COMPLETED
    
    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        等待任务完成并返回结果
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
        
        Returns:
            包含完整结果的字典，见 run() 方法返回值
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('复杂任务', wait=False)
            >>> # 做一些其他事情...
            >>> result = runner.wait(timeout=60)
            >>> print(result['output'])
        """
        logger.info(f"等待任务完成, 超时: {timeout}")
        
        if self._thread:
            self._thread.join(timeout=timeout)
        
        # 收集输出
        output_parts = []
        while not self._output_queue.empty():
            item = self._output_queue.get_nowait()
            if isinstance(item, tuple) and item[0] == "text":
                output_parts.append(item[1])
        
        output_text = "".join(output_parts)
        
        # 检查是否有错误
        if self._state == RunState.ERROR:
            for event in self.full_json_events:
                if event.get("type") == "error":
                    output_text = f"Error: {event.get('error', {})}"
                    break
        
        # 记录助手回复
        self.conversation_history.append({
            "role": "assistant",
            "content": output_text,
            "json_events": [e for e in self.full_json_events if e.get("type") == "text"],
        })
        
        logger.info(f"任务结束，状态: {self._state.value}")
        
        return {
            "output": output_text,
            "session_id": self.session_id,
            "task_dir": self.task_dir,
            "state": self._state.value,
            "json_events": self.full_json_events,
        }
    
    def get_state(self) -> RunState:
        """
        获取当前运行状态
        
        Returns:
            RunState 枚举值
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('任务', wait=False)
            >>> print(runner.get_state())  # RunState.RUNNING
        """
        return self._state
    
    def get_session_id(self) -> Optional[str]:
        """
        获取当前会话 ID
        
        Returns:
            会话 ID 字符串，如果还没有则返回 None
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('Hello')
            >>> print(runner.get_session_id())  # 'ses_xxx...'
        """
        return self.session_id
    
    def get_current_output(self) -> str:
        """
        获取当前已产生的输出（实时获取）
        
        Returns:
            截止到当前时刻已生成的输出文本
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('生成一首长诗', wait=False)
            >>> time.sleep(2)
            >>> print(runner.get_current_output())  # 部分输出
        """
        output_parts = []
        while not self._output_queue.empty():
            item = self._output_queue.get_nowait()
            if isinstance(item, tuple) and item[0] == "text":
                output_parts.append(item[1])
        return "".join(output_parts)
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的 JSON 事件
        
        Args:
            count: 返回的事件数量，默认 10
        
        Returns:
            JSON 事件列表
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('任务', wait=False)
            >>> time.sleep(1)
            >>> events = runner.get_recent_events(5)
            >>> for e in events:
            ...     print(e.get('type'))
        """
        return self.full_json_events[-count:]
    
    def is_running(self) -> bool:
        """
        检查任务是否仍在运行
        
        Returns:
            True 表示正在运行，False 表示已结束
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('任务', wait=False)
            >>> while runner.is_running():
            ...     time.sleep(1)
        """
        return self._state == RunState.RUNNING
    
    def interrupt(self) -> bool:
        """
        发送中断信号（SIGINT）尝试终止任务
        
        这是优雅的终止方式，进程可以清理资源。
        
        Returns:
            True 表示成功发送中断信号，False 表示进程已结束
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('长时间任务', wait=False)
            >>> time.sleep(5)
            >>> if runner.interrupt():
            ...     print('已发送中断信号')
        """
        if self._process and self._process.poll() is None:
            logger.info("发送中断信号 (SIGINT)")
            self._process.send_signal(signal.SIGINT)
            self._state = RunState.INTERRUPTED
            return True
        logger.warning("无法中断：进程已结束")
        return False
    
    def kill(self) -> bool:
        """
        强制终止任务（SIGKILL）
        
        这会立即终止进程，不会进行清理。
        只有在 interrupt() 无效时使用。
        
        Returns:
            True 表示成功终止，False 表示进程已结束
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('任务', wait=False)
            >>> time.sleep(5)
            >>> runner.kill()  # 强制终止
        """
        if self._process and self._process.poll() is None:
            logger.warning("强制终止进程 (SIGKILL)")
            self._process.kill()
            self._state = RunState.INTERRUPTED
            return True
        logger.warning("无法终止：进程已结束")
        return False
    
    def save_markdown(self, filepath: Optional[str] = None) -> str:
        """
        将对话保存为 Markdown 文件
        
        Args:
            filepath: 保存路径，默认保存到任务目录下的 conversation.md
        
        Returns:
            实际保存的文件路径
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('问题1')
            >>> runner.run('问题2')
            >>> path = runner.save_markdown()
            >>> print(f'已保存到: {path}')
            
            >>> # 指定自定义路径
            >>> path = runner.save_markdown('/path/to/save.md')
        """
        if filepath is None:
            filepath = os.path.join(self.task_dir, "conversation.md")
        
        with open(filepath, "w", encoding="utf-8") as f:
            # 写入标题和元信息
            f.write(f"# {self.task_name}\n\n")
            f.write(f"**Model**: {self.model or 'default'}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Task Dir**: {self.task_dir}\n\n")
            f.write(f"**State**: {self._state.value}\n\n")
            
            if self.session_id:
                f.write(f"**Session ID**: {self.session_id}\n\n")
            
            f.write("---\n\n")
            
            # 写入对话历史
            for msg in self.conversation_history:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    f.write(f"## User\n\n{content}\n\n")
                else:
                    f.write(f"## Assistant\n\n{content}\n\n")
        
        logger.info(f"对话已保存到: {filepath}")
        return filepath
    
    def get_conversation_markdown(self) -> str:
        """
        获取对话的 Markdown 格式字符串
        
        Returns:
            Markdown 格式的对话内容
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('Hello')
            >>> md = runner.get_conversation_markdown()
            >>> print(md)
        """
        lines = [
            f"# {self.task_name}\n\n",
            f"**Model**: {self.model or 'default'}\n\n",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"**Task Dir**: {self.task_dir}\n\n",
            f"**State**: {self._state.value}\n\n",
        ]
        
        if self.session_id:
            lines.append(f"**Session ID**: {self.session_id}\n\n")
        
        lines.append("---\n\n")
        
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                lines.append(f"## User\n\n{content}\n\n")
            else:
                lines.append(f"## Assistant\n\n{content}\n\n")
        
        return "".join(lines)
    
    def get_json_events(self) -> List[Dict[str, Any]]:
        """
        获取所有 JSON 事件
        
        Returns:
            包含所有 JSON 事件的列表
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('任务')
            >>> events = runner.get_json_events()
            >>> print(f"共 {len(events)} 个事件")
        """
        return self.full_json_events
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表，每条包含 role, content, context 等字段
        
        Example:
            >>> runner = OpenCodeRunner(task_name='test')
            >>> runner.run('问题1')
            >>> runner.run('问题2')
            >>> history = runner.get_conversation_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content'][:30]}...")
        """
        return self.conversation_history


def main():
    """
    命令行入口点
    
    提供命令行界面来使用 OpenCodeRunner 的功能。
    """
    import argparse
    import time
    import json
    
    parser = argparse.ArgumentParser(
        description="OpenCode Python Runner - 通过 Python 驱动 opencode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认加载当前目录的 .env 文件
  python opencode_runner.py --task-name test --message "say hello"
  
  # 指定配置文件
  python opencode_runner.py --task-name test --message "say hello" --config prod.env
  
  # 使用代码设置模型源
  python opencode_runner.py --task-name mytask --model opencode/claude-sonnet-4-5 --message "help me"
  python opencode_runner.py --task-name test --message "task" --no-wait
  python opencode_runner.py --task-name test --message "写代码" --model-source-type code
        """
    )
    parser.add_argument("--task-name", required=True, help="任务名称")
    parser.add_argument("--model", help="使用的模型 (如 opencode/claude-sonnet-4-5)")
    parser.add_argument("--context", help="初始上下文/系统提示")
    parser.add_argument("--workspace", help="工作空间目录")
    parser.add_argument("--message", required=True, help="发送给 opencode 的消息")
    parser.add_argument("--save-markdown", nargs="?", const="auto", 
                       help="保存对话到 markdown 文件")
    parser.add_argument("--no-wait", action="store_true", 
                       help="不等待完成，允许实时监控")
    parser.add_argument("--model-source-type", 
                       choices=["plan", "think", "code", "write", "default"],
                       help="指定使用哪个场景的模型")
    parser.add_argument("--config", 
                       help="模型源配置文件路径 (.env 格式)，默认加载当前目录的 .env")
    parser.add_argument("--no-env", action="store_true",
                       help="不加载默认的 .env 配置文件")
    
    args = parser.parse_args()
    
    # 加载模型源配置
    model_source = None
    
    # 先尝试加载默认的 .env 文件
    if not args.no_env:
        default_env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(default_env_path):
            model_source = ModelSource.from_env_file(default_env_path)
            logger.info(f"已加载默认 .env 配置: {default_env_path}")
    
    # 如果指定了配置文件，则加载并覆盖
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)
        model_source = ModelSource.from_env_file(config_path)
        logger.info(f"已加载配置文件: {config_path}")
    
    # 创建运行器
    runner = OpenCodeRunner(
        task_name=args.task_name,
        model=args.model,
        context=args.context,
        workspace=args.workspace,
        model_source=model_source,
    )
    
    print(f"任务目录: {runner.task_dir}")
    print(f"正在启动任务...\n")
    
    # 确定模型源类型
    model_source_type = None
    if args.model_source_type:
        type_mapping = {
            "plan": ModelSourceType.PLAN,
            "think": ModelSourceType.THINK,
            "code": ModelSourceType.CODE,
            "write": ModelSourceType.WRITE,
            "default": ModelSourceType.DEFAULT,
        }
        model_source_type = type_mapping[args.model_source_type]
    
    # 执行任务
    result = runner.run(
        args.message, 
        args.context, 
        wait=not args.no_wait,
        model_source_type=model_source_type,
    )
    
    if args.no_wait:
        # 监控模式
        print(f"状态: {result['state']}")
        print("监控中... (按 Ctrl+C 中断)")
        
        try:
            while runner.is_running():
                time.sleep(1)
                current = runner.get_current_output()
                if current:
                    print(f"\n--- 当前输出 ---\n{current}\n")
        except KeyboardInterrupt:
            print("\n正在中断...")
            runner.interrupt()
            time.sleep(1)
        
        result = runner.wait()
    
    print(f"\n会话 ID: {result['session_id']}")
    print(f"最终状态: {result['state']}")
    print(f"\n输出:\n{result['output']}")
    
    # 保存 markdown
    if args.save_markdown:
        filepath = runner.save_markdown() if args.save_markdown == "auto" else args.save_markdown
        print(f"\n已保存到: {filepath}")


if __name__ == "__main__":
    main()
