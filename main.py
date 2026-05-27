# ============================================================
# 0. Imports
# ============================================================

from __future__ import annotations

import os
import re
import time
import math
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import pandas as pd


# ============================================================
# 1. Path & Experiment Configuration
# ============================================================

project_root_str = r"D:\remote_py_code\MCTS_new"

PROJECT_ROOT = Path(project_root_str).expanduser().resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# OpenAI-compatible API settings.
# 这里使用 DeepSeek 的 OpenAI 兼容接口。
os.environ["OPENAI_API_KEY"] = "xx"  # deepseek
model_name = "deepseek-v4-pro"
base_url = "https://api.deepseek.com"


LLMBackend = Literal["local", "openai"]
OpenAIMode = Literal["chat", "responses"]
ChatMessage = Dict[str, str]


@dataclass
class LLMUsage:
    """一次 LLM 调用的 token 使用量。

    本地模型通过 tokenizer 估算；API 模型优先读取服务端返回的 usage。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """一次 LLM 调用的结构化返回结果。"""

    content: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    """保存到节点中的 LLM 调用记录。"""

    purpose: str
    usage: LLMUsage
    elapsed_seconds: float
    error: Optional[str] = None

    @classmethod
    def from_response(cls, purpose: str, response: LLMResponse) -> "LLMCallRecord":
        return cls(
            purpose=purpose,
            usage=response.usage,
            elapsed_seconds=response.elapsed_seconds,
            error=response.error,
        )


@dataclass
class RewardResult:
    """一次奖励评估的结构化结果。"""

    score: float
    call_record: Optional[LLMCallRecord] = None


@dataclass
class ExperimentConfig:
    """实验级配置。

    该对象只保存静态参数，不承载运行状态。
    """

    experiment_name: str = "cwe_119"
    data_csv: Path = PROJECT_ROOT / "data" / "few_test_set_cwe_119.csv"
    output_root: Path = OUTPUTS_DIR

    # 数据字段。
    code_column: str = "func_before"
    label_column: str = "vul"
    id_column: str = "Unnamed: 0"

    # 样本范围。与原始代码 range(88, 92) 保持一致。
    index_start: int = 0
    index_end: int = 92

    # 输入长度与搜索参数。
    max_function_chars: int = 9000
    max_iter: int = 4
    max_expand: int = 3
    exploration_c: float = 1.4

    # LLM 后端。
    llm_backend: LLMBackend = "local"

    # 本地模型配置。
    cuda_visible_devices: str = "0"
    local_model_name_or_path: str = "/home/cd/t1/modelscope_download/Qwen3-8B"
    local_torch_dtype: str = "auto"
    local_device_map: str = "auto"
    local_enable_thinking: bool = False
    qwen_think_end_token_id: int = 151668

    # OpenAI API 配置。
    openai_model: str = "gpt-4.1-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url: Optional[str] = None
    openai_mode: OpenAIMode = "chat"

    # 生成配置。
    max_new_tokens: int = 32768
    temperature: float = 0.7
    reward_temperature: float = 0.1
    timeout: int = 360

    # 本地临时 LLM 对话缓存。
    # 缓存键由模型身份、调用参数和标准 messages 共同决定。
    use_llm_cache: bool = True
    llm_cache_dir: Path = OUTPUTS_DIR / "_llm_cache"

    @property
    def experiment_output_dir(self) -> Path:
        return self.output_root / self.experiment_name


# ============================================================
# 2. LLM Backend Abstraction
# ============================================================


class BaseLLMClient:
    """统一 LLM 调用接口。

    LLMClient 只负责接收外部已经组装好的 messages，并执行底层模型/API调用。
    对话历史的维护、裁剪、拼接与保存，均由上层搜索过程负责。
    """

    def generate(
        self,
        messages: List[ChatMessage],
        timeout: int = 360,
        temperature: float = 0.7,
    ) -> LLMResponse:
        raise NotImplementedError


class CachedLLMClient(BaseLLMClient):
    """LLM 调用缓存包装器。

    该类不改变底层客户端职责。底层客户端仍然只负责真实模型调用；
    本类负责在调用前检查本地缓存，并在未命中时写入缓存。
    """

    def __init__(
        self,
        inner_client: BaseLLMClient,
        cache_dir: Path,
        cache_identity: str,
        enabled: bool = True,
    ) -> None:
        self.inner_client = inner_client
        self.cache_dir = Path(cache_dir)
        self.cache_identity = cache_identity
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        messages: List[ChatMessage],
        timeout: int = 360,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if not self.enabled:
            return self.inner_client.generate(
                messages=messages,
                timeout=timeout,
                temperature=temperature,
            )

        time0 = time.perf_counter()
        cache_key = self.build_cache_key(
            messages=messages,
            temperature=temperature,
        )
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf8") as f:
                    cached_payload = json.load(f)
                elapsed_seconds = time.perf_counter() - time0
                response = self.response_from_cache_payload(
                    cached_payload=cached_payload,
                    cache_key=cache_key,
                    elapsed_seconds=elapsed_seconds,
                )
                print(f"cache hit! time taken: {elapsed_seconds:.4f} seconds.")
                return response
            except Exception as e:
                print(f"[LLM Cache Read Error]: {e}. Fallback to live call.")

        response = self.inner_client.generate(
            messages=messages,
            timeout=timeout,
            temperature=temperature,
        )
        if response.error is None:
            self.write_cache(
                cache_path=cache_path,
                cache_key=cache_key,
                messages=messages,
                temperature=temperature,
                response=response,
            )
        return response

    def build_cache_key(self, messages: List[ChatMessage], temperature: float) -> str:
        payload = {
            "cache_identity": self.cache_identity,
            "temperature": temperature,
            "messages": messages,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf8")).hexdigest()

    def write_cache(
        self,
        cache_path: Path,
        cache_key: str,
        messages: List[ChatMessage],
        temperature: float,
        response: LLMResponse,
    ) -> None:
        payload = {
            "cache_key": cache_key,
            "cache_identity": self.cache_identity,
            "temperature": temperature,
            "messages": messages,
            "response": self.response_to_cache_dict(response),
        }
        try:
            with cache_path.open("w", encoding="utf8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[LLM Cache Write Error]: {e}")

    @staticmethod
    def response_to_cache_dict(response: LLMResponse) -> Dict[str, Any]:
        return {
            "content": response.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "raw": response.usage.raw,
            },
            "elapsed_seconds": response.elapsed_seconds,
            "error": response.error,
        }

    @staticmethod
    def response_from_cache_payload(
        cached_payload: Dict[str, Any],
        cache_key: str,
        elapsed_seconds: float,
    ) -> LLMResponse:
        cached_response = cached_payload.get("response", {})
        cached_usage = cached_response.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            raw={
                "cache_hit": True,
                "cache_key": cache_key,
                "cache_identity": cached_payload.get("cache_identity"),
                "cached_response_usage": cached_usage,
            },
        )
        return LLMResponse(
            content=cached_response.get("content", ""),
            usage=usage,
            elapsed_seconds=elapsed_seconds,
            error=cached_response.get("error"),
        )


class LocalTransformersClient(BaseLLMClient):
    """本地 transformers 模型客户端。"""

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 32768,
        enable_thinking: bool = False,
        think_end_token_id: int = 151668,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.think_end_token_id = think_end_token_id

    def generate(
        self,
        messages: List[ChatMessage],
        timeout: int = 360,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """调用本地模型生成回复。

        timeout 参数保留用于接口兼容，本地 transformers generate 不直接使用该参数。
        """
        time0 = time.perf_counter()

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            prompt_tokens = int(model_inputs.input_ids.shape[-1])

            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
            }
            if temperature and temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["do_sample"] = True
            else:
                generation_kwargs["do_sample"] = False

            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            completion_tokens = len(output_ids)

            final_content = self._decode_final_content(output_ids)
            answer = final_content.replace("**", "").strip()
            elapsed_seconds = time.perf_counter() - time0

            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                raw={
                    "backend": "local",
                    "estimated": True,
                },
            )

            print(f"response received! time taken: {elapsed_seconds:.2f} seconds.")
            self.torch.cuda.empty_cache()
            return LLMResponse(content=answer, usage=usage, elapsed_seconds=elapsed_seconds)

        except Exception as e:
            elapsed_seconds = time.perf_counter() - time0
            error_message = f"[Local Inference Error]: {e}"
            print(error_message)
            return LLMResponse(content=error_message, elapsed_seconds=elapsed_seconds, error=error_message)

    def _decode_final_content(self, output_ids: List[int]) -> str:
        """解析 Qwen3 可能包含的 think 内容，只返回最终回复。"""
        try:
            think_end_index = len(output_ids) - output_ids[::-1].index(self.think_end_token_id)
        except ValueError:
            think_end_index = 0

        final_ids = output_ids[think_end_index:]
        return self.tokenizer.decode(final_ids, skip_special_tokens=True).strip()


class OpenAIAPIClient(BaseLLMClient):
    """OpenAI API 客户端。

    openai_mode='chat' 使用 Chat Completions API。
    openai_mode='responses' 使用 Responses API。
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        mode: OpenAIMode = "chat",
        max_output_tokens: int = 32768,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing OpenAI API key environment variable: {api_key_env}")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.mode = mode
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        messages: List[ChatMessage],
        timeout: int = 360,
        temperature: float = 0.7,
    ) -> LLMResponse:
        time0 = time.perf_counter()

        try:
            if self.mode == "responses":
                answer, usage = self._generate_with_responses(
                    messages=messages,
                    timeout=timeout,
                    temperature=temperature,
                )
            else:
                answer, usage = self._generate_with_chat_completions(
                    messages=messages,
                    timeout=timeout,
                    temperature=temperature,
                )

            answer = answer.replace("**", "").strip()
            elapsed_seconds = time.perf_counter() - time0
            print(f"response received! time taken: {elapsed_seconds:.2f} seconds.")
            return LLMResponse(content=answer, usage=usage, elapsed_seconds=elapsed_seconds)

        except Exception as e:
            elapsed_seconds = time.perf_counter() - time0
            error_message = f"[OpenAI API Error]: {e}"
            print(error_message)
            return LLMResponse(content=error_message, elapsed_seconds=elapsed_seconds, error=error_message)

    def _generate_with_chat_completions(
        self,
        messages: List[ChatMessage],
        timeout: int,
        temperature: float,
    ) -> tuple[str, LLMUsage]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            max_tokens=self.max_output_tokens,
        )
        usage = self._parse_openai_usage(getattr(completion, "usage", None))
        return completion.choices[0].message.content or "", usage

    def _generate_with_responses(
        self,
        messages: List[ChatMessage],
        timeout: int,
        temperature: float,
    ) -> tuple[str, LLMUsage]:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            temperature=temperature,
            timeout=timeout,
            max_output_tokens=self.max_output_tokens,
        )
        usage = self._parse_openai_usage(getattr(response, "usage", None))
        return getattr(response, "output_text", "") or "", usage

    @staticmethod
    def _parse_openai_usage(raw_usage: Any) -> LLMUsage:
        if raw_usage is None:
            return LLMUsage(raw={"backend": "openai", "usage_available": False})

        def read_field(obj: Any, *names: str) -> int:
            for name in names:
                value = getattr(obj, name, None)
                if value is not None:
                    return int(value)
                if isinstance(obj, dict) and name in obj:
                    return int(obj[name])
            return 0

        prompt_tokens = read_field(raw_usage, "prompt_tokens", "input_tokens")
        completion_tokens = read_field(raw_usage, "completion_tokens", "output_tokens")
        total_tokens = read_field(raw_usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        try:
            raw = raw_usage.model_dump()
        except Exception:
            raw = dict(raw_usage) if isinstance(raw_usage, dict) else {"repr": repr(raw_usage)}
        raw["backend"] = "openai"

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=raw,
        )


# ============================================================
# 3. Prompt Builders
# ============================================================


class VulnerabilityPromptBuilder:
    """集中维护所有 Prompt。"""

    @staticmethod
    def build_initial_analysis_prompt(function_code: str) -> str:
        return f"""
You are a world-class cybersecurity expert specializing in static code analysis. Your task is to analyze the following code function for any potential vulnerabilities.

Please provide your analysis in the following strict format. The Vulnerability Status MUST be one of: Vulnerable / Not Vulnerable.
[Reasoning]: (A step-by-step explanation of why it is or is not a vulnerability)
[Vulnerability Status]: (Vulnerable / Not Vulnerable)
[Vulnerability Type]: (e.g., Command Injection, SQL Injection, Not Applicable)

Here is the function to analyze:
```python
{function_code}
```
Let's think step by step.
"""

    @staticmethod
    def build_feedback_prompt(function_code: str, weak_answer_report: str) -> str:
        return f"""
You are a senior code reviewer reviewing an initial security analysis of a function. The initial analysis might be flawed (e.g., missed a vulnerability, identified a non-existent one, or misunderstood the root cause).
Function Code:
```python
{function_code}
```
Initial Analysis Report:
```
{weak_answer_report}
```
Your task is to critically analyze the initial report and provide feedback. Point out the flaws in its logic. What clues in the code were overlooked? What security principles were misapplied? Provide a reflection that can guide the analyst to a better conclusion. Let's think step by step.
"""

    @staticmethod
    def build_refined_analysis_prompt(
        function_code: str,
        weak_answer_report: str,
        security_feedback_hints: str,
    ) -> str:
        return f"""
You are a security analyst who has received feedback on your initial analysis. Please refine your report based on the feedback provided.
Function Code:
```python
{function_code}
```
Your Initial (Flawed) Analysis:
```{weak_answer_report}
```
Feedback from Senior Reviewer:
```
{security_feedback_hints}
```
Now, please provide a new, corrected, and more accurate vulnerability analysis report in the following strict format. The Vulnerability Status MUST be one of: Vulnerable / Not Vulnerable.
[Reasoning]: (A step-by-step explanation of why it is or is not a vulnerability)
[Vulnerability Status]: (Vulnerable / Not Vulnerable)
[Vulnerability Type]: (e.g., Command Injection, SQL Injection, Not Applicable)
"""

    @staticmethod
    def build_self_critique_prompt(function_code: str, generated_report_str: str) -> str:
        return f"""
You are an aggressive 'red team' security auditor. Your goal is to challenge every assumption in the following security analysis report for the given function. Find any possible flaw, logical inconsistency, or overlooked detail. Be extremely critical.

The function under review:
```python
{function_code}
```

The analysis report to critique:
```
{generated_report_str}
```

After your critical analysis, you MUST conclude with a "Logical Consistency Score". This score, from -100 (completely flawed) to +100 (logically sound and unassailable), represents your confidence in the original report's conclusion after your rigorous critique.

Your response format MUST be:
[Critique]: ... (Your detailed critique here)
[Logical Consistency Score]: ... (A single number between -100 and 100)
"""


# ============================================================
# 4. Report Parsing & Label Conversion
# ============================================================


@dataclass
class ParsedVulnerabilityReport:
    vulnerability_status: str = "Not Parsable"
    vulnerability_type: str = "Not Parsable"
    reasoning: str = "Not Parsable"

    @property
    def is_parsable(self) -> bool:
        return all(
            value != "Not Parsable"
            for value in [self.vulnerability_status, self.vulnerability_type, self.reasoning]
        )


class VulnerabilityReportParser:
    """漏洞报告解析器。"""

    STATUS_PATTERN = re.compile(
        r"\[?\s*Vulnerability Status\]?\s*:\s*(Vulnerable|Not Vulnerable)",
        re.IGNORECASE,
    )
    TYPE_PATTERN = re.compile(
        r"\[?\s*Vulnerability Type\]?\s*:\s*(.*)",
        re.IGNORECASE,
    )
    REASONING_PATTERN = re.compile(
        r"\[?\s*Reasoning\]?\s*:\s*((?:.|\n)*?)(?=\n\s*\[?\s*Vulnerability Status\]?\s*:|\n\s*\[?\s*Vulnerability Type\]?\s*:|$)",
        re.IGNORECASE,
    )
    SCORE_PATTERN = re.compile(
        r"\[?\s*Logical Consistency Score\]?\s*:\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    def parse_report(self, report_str: str) -> ParsedVulnerabilityReport:
        parsed = ParsedVulnerabilityReport()

        status_match = self.STATUS_PATTERN.search(report_str)
        if status_match:
            parsed.vulnerability_status = status_match.group(1).strip()

        type_match = self.TYPE_PATTERN.search(report_str)
        if type_match:
            parsed.vulnerability_type = type_match.group(1).strip()

        reasoning_match = self.REASONING_PATTERN.search(report_str)
        if reasoning_match:
            parsed.reasoning = reasoning_match.group(1).strip()

        return parsed

    def parse_logical_consistency_score(self, critique_response: str) -> float:
        score_match = self.SCORE_PATTERN.search(critique_response)
        if not score_match:
            return 0.0
        try:
            return float(score_match.group(1))
        except ValueError:
            return 0.0


def status_to_label(status: str) -> int:
    """将 Vulnerability Status 转换为二分类标签。

    1: Vulnerable
    0: Not Vulnerable
    -1: 无法判断
    """
    normalized = status.strip().lower()
    if normalized == "not vulnerable":
        return 0
    if normalized == "vulnerable":
        return 1
    return -1


def is_llm_error(text: str) -> bool:
    return "[Local Inference Error]" in text or "[OpenAI API Error]" in text


# ============================================================
# 5. MCTS Data Structures
# ============================================================


@dataclass
class MCTSNode:
    """MCTS 树中的一个节点。

    每个节点表示一次漏洞分析报告，并保存生成该报告的标准对话历史。
    history 始终使用 [{"role": xx, "content": xx}] 格式。
    如需其他形式，应从该标准格式派生。
    """

    node_id: int
    report: str
    history: List[ChatMessage]
    generation_call_records: List[LLMCallRecord] = field(default_factory=list)
    reward_call_records: List[LLMCallRecord] = field(default_factory=list)
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    q_value: float = 0.0
    ucb_value: float = float("-inf")
    visit_count: int = 0
    depth: int = 0

    def add_reward(self, reward: float) -> None:
        self.rewards.append(reward)
        self.visit_count = len(self.rewards)
        self.q_value = float(np.mean(self.rewards)) if self.rewards else 0.0

    @staticmethod
    def _sum_usage(records: List[LLMCallRecord]) -> LLMUsage:
        prompt_tokens = sum(record.usage.prompt_tokens for record in records)
        completion_tokens = sum(record.usage.completion_tokens for record in records)
        total_tokens = sum(record.usage.total_tokens for record in records)
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw={"aggregated": True},
        )

    @property
    def generation_usage(self) -> LLMUsage:
        return self._sum_usage(self.generation_call_records)

    @property
    def reward_usage(self) -> LLMUsage:
        return self._sum_usage(self.reward_call_records)

    @property
    def generation_elapsed_seconds(self) -> float:
        return sum(record.elapsed_seconds for record in self.generation_call_records)

    @property
    def reward_elapsed_seconds(self) -> float:
        return sum(record.elapsed_seconds for record in self.reward_call_records)


@dataclass
class SearchResult:
    nodes: Dict[int, MCTSNode]
    root_id: Optional[int]
    best_node_id: Optional[int]
    final_prediction: int
    error: Optional[str] = None

    @property
    def best_report(self) -> Optional[str]:
        if self.best_node_id is None:
            return None
        node = self.nodes.get(self.best_node_id)
        return node.report if node else None


# ============================================================
# 6. Reward Evaluator
# ============================================================


class SelfCritiqueRewardEvaluator:
    """无监督自我批判奖励评估器。

    注意：每次 evaluate() 都会在外部组装 messages 后触发一次 LLM 调用。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_builder: VulnerabilityPromptBuilder,
        parser: VulnerabilityReportParser,
        reward_temperature: float = 0.1,
        timeout: int = 360,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.reward_temperature = reward_temperature
        self.timeout = timeout

    def evaluate(self, function_code: str, generated_report_str: str) -> RewardResult:
        parsed_report = self.parser.parse_report(generated_report_str)

        if is_llm_error(generated_report_str):
            return RewardResult(score=-100.0)
        if not parsed_report.is_parsable:
            return RewardResult(score=-100.0)
        if len(parsed_report.reasoning) < 50:
            return RewardResult(score=-20.0)

        critique_prompt = self.prompt_builder.build_self_critique_prompt(
            function_code=function_code,
            generated_report_str=generated_report_str,
        )
        messages = [{"role": "user", "content": critique_prompt}]
        response = self.llm_client.generate(
            messages,
            temperature=self.reward_temperature,
            timeout=self.timeout,
        )
        score = self.parser.parse_logical_consistency_score(response.content)
        return RewardResult(
            score=score,
            call_record=LLMCallRecord.from_response("self_critique_reward", response),
        )


# ============================================================
# 7. MCTS Search Process
# ============================================================


class VulnerabilityMCTSSearch:
    """一次函数漏洞分析的 MCTS 搜索过程。

    该对象维护完整搜索树，包括节点、父子关系、奖励、Q 值、UCB 值与生成历史。
    """

    def __init__(
        self,
        function_code: str,
        llm_client: BaseLLMClient,
        prompt_builder: VulnerabilityPromptBuilder,
        parser: VulnerabilityReportParser,
        reward_evaluator: SelfCritiqueRewardEvaluator,
        writer: Optional["ResultWriter"] = None,
        sample_index: Optional[int] = None,
        max_iter: int = 4,
        max_expand: int = 3,
        exploration_c: float = 1.4,
        generation_temperature: float = 0.7,
        timeout: int = 360,
    ) -> None:
        self.function_code = function_code
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.reward_evaluator = reward_evaluator
        self.writer = writer
        self.sample_index = sample_index

        self.max_iter = max_iter
        self.max_expand = max_expand
        self.exploration_c = exploration_c
        self.generation_temperature = generation_temperature
        self.timeout = timeout

        self.nodes: Dict[int, MCTSNode] = {}
        self.report_to_node_id: Dict[str, int] = {}
        self.root_id: Optional[int] = None
        self.next_node_id: int = 0
        self.counter: int = 0

    def run(self) -> SearchResult:
        if self.writer and self.sample_index is not None:
            self.writer.init_prediction_csv(self.sample_index)

        root_id = self.initialize_root()
        if root_id is None:
            return SearchResult(
                nodes=self.nodes,
                root_id=None,
                best_node_id=None,
                final_prediction=-1,
                error="Initial analysis failed.",
            )

        self.evaluate_node(root_id)
        self.update_tree_statistics()
        self.save_iteration_state()

        for i in range(self.max_iter):
            self.counter += 1
            print(f"\n======= Iteration {i + 1}/{self.max_iter} =======")

            selected_node_id = self.select_node()
            if selected_node_id is None:
                break

            new_node_id = self.expand_node(selected_node_id)
            if new_node_id is not None:
                self.evaluate_node(new_node_id)

            self.update_tree_statistics()
            self.save_iteration_state()

        best_node_id = self.get_best_node_id()
        final_prediction = self.get_final_prediction(best_node_id)
        return SearchResult(
            nodes=self.nodes,
            root_id=self.root_id,
            best_node_id=best_node_id,
            final_prediction=final_prediction,
            error=None,
        )

    def initialize_root(self) -> Optional[int]:
        initial_prompt = self.prompt_builder.build_initial_analysis_prompt(self.function_code)
        initial_messages = self.build_messages(history=[], user_prompt=initial_prompt)
        initial_response = self.llm_client.generate(
            initial_messages,
            temperature=self.generation_temperature,
            timeout=self.timeout,
        )
        initial_report = initial_response.content
        root_history = self.append_assistant_message(initial_messages, initial_report)

        self.save_report(report=initial_report, history=root_history)

        if is_llm_error(initial_report):
            print("Initial analysis failed due to LLM error. Aborting.")
            return None

        root_id = self.create_node(
            report=initial_report,
            history=root_history,
            generation_call_records=[LLMCallRecord.from_response("initial_analysis", initial_response)],
            parent_id=None,
        )
        self.root_id = root_id
        return root_id

    @staticmethod
    def build_messages(history: List[ChatMessage], user_prompt: str) -> List[ChatMessage]:
        """由标准 history 派生出本次待发送给 LLMClient 的 messages。"""
        return list(history) + [{"role": "user", "content": user_prompt}]

    @staticmethod
    def append_assistant_message(messages: List[ChatMessage], answer: str) -> List[ChatMessage]:
        """将 LLM 返回结果写回标准 history。"""
        return list(messages) + [{"role": "assistant", "content": answer}]

    def create_node(
        self,
        report: str,
        history: List[ChatMessage],
        generation_call_records: List[LLMCallRecord],
        parent_id: Optional[int],
    ) -> int:
        """创建节点。

        如果完全相同的报告已经存在，则返回已有节点编号。
        """
        if report in self.report_to_node_id:
            return self.report_to_node_id[report]

        node_id = self.next_node_id
        self.next_node_id += 1

        parent_depth = self.nodes[parent_id].depth if parent_id is not None else -1
        node = MCTSNode(
            node_id=node_id,
            report=report,
            history=list(history),
            generation_call_records=list(generation_call_records),
            parent_id=parent_id,
            depth=parent_depth + 1,
        )
        self.nodes[node_id] = node
        self.report_to_node_id[report] = node_id

        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)

        return node_id

    def evaluate_node(self, node_id: int) -> float:
        node = self.nodes[node_id]
        reward_result = self.reward_evaluator.evaluate(self.function_code, node.report)
        node.add_reward(reward_result.score)
        if reward_result.call_record is not None:
            node.reward_call_records.append(reward_result.call_record)
        return reward_result.score

    def select_node(self) -> Optional[int]:
        candidates = self.filter_mature_nodes()
        if not candidates:
            return None

        best_node_id = None
        highest_ucb = float("-inf")
        for node_id in candidates:
            node = self.nodes[node_id]
            if node.ucb_value > highest_ucb:
                highest_ucb = node.ucb_value
                best_node_id = node_id
        return best_node_id

    def filter_mature_nodes(self) -> List[int]:
        """筛选仍值得扩展的节点。

        保持原始逻辑：
        1. 子节点数量未达到 max_expand；或
        2. 当前节点的平均奖励仍高于所有子节点。
        """
        filtered = []
        for node_id, node in self.nodes.items():
            if not node.rewards:
                continue

            cond1 = len(node.children_ids) < self.max_expand
            child_q_values = [self.nodes[child_id].q_value for child_id in node.children_ids]
            cond2 = max(child_q_values) < node.q_value if child_q_values else True

            if cond1 or cond2:
                filtered.append(node_id)

        return filtered

    def expand_node(self, node_id: int) -> Optional[int]:
        parent_node = self.nodes[node_id]

        feedback_prompt = self.prompt_builder.build_feedback_prompt(
            function_code=self.function_code,
            weak_answer_report=parent_node.report,
        )
        feedback_messages = self.build_messages(
            history=parent_node.history,
            user_prompt=feedback_prompt,
        )
        feedback_response = self.llm_client.generate(
            feedback_messages,
            temperature=self.generation_temperature,
            timeout=self.timeout,
        )
        feedback = feedback_response.content
        feedback_history = self.append_assistant_message(feedback_messages, feedback)

        refined_prompt = self.prompt_builder.build_refined_analysis_prompt(
            function_code=self.function_code,
            weak_answer_report=parent_node.report,
            security_feedback_hints=feedback,
        )
        refined_messages = self.build_messages(
            history=feedback_history,
            user_prompt=refined_prompt,
        )
        refined_response = self.llm_client.generate(
            refined_messages,
            temperature=self.generation_temperature,
            timeout=self.timeout,
        )
        new_report = refined_response.content
        child_messages = self.append_assistant_message(refined_messages, new_report)

        self.save_report(report=new_report, history=child_messages)

        if is_llm_error(new_report):
            return None

        generation_call_records = list(parent_node.generation_call_records) + [
            LLMCallRecord.from_response("security_feedback", feedback_response),
            LLMCallRecord.from_response("refined_analysis", refined_response),
        ]
        new_node_id = self.create_node(
            report=new_report,
            history=child_messages,
            generation_call_records=generation_call_records,
            parent_id=node_id,
        )
        return new_node_id

    def update_tree_statistics(self) -> None:
        for node in self.nodes.values():
            if not node.rewards:
                continue
            parent_visits = 0
            if node.parent_id is not None:
                parent_visits = self.nodes[node.parent_id].visit_count
            node.ucb_value = self.compute_ucb(
                q_value=node.q_value,
                parent_visit_count=parent_visits,
                node_visit_count=node.visit_count,
                c=self.exploration_c,
            )

    @staticmethod
    def compute_ucb(
        q_value: float,
        parent_visit_count: int,
        node_visit_count: int,
        c: float,
    ) -> float:
        return q_value + c * math.sqrt(
            math.log(parent_visit_count + 1) / (node_visit_count + 1e-5)
        )

    def get_best_node_id(self) -> Optional[int]:
        valid_nodes = [node for node in self.nodes.values() if node.rewards]
        if not valid_nodes:
            return None
        best_node = max(valid_nodes, key=lambda node: node.q_value)
        return best_node.node_id

    def get_final_prediction(self, best_node_id: Optional[int]) -> int:
        if best_node_id is None:
            return -1
        best_node = self.nodes[best_node_id]
        parsed_report = self.parser.parse_report(best_node.report)
        return status_to_label(parsed_report.vulnerability_status)

    def save_report(self, report: str, history: List[ChatMessage]) -> None:
        if self.writer and self.sample_index is not None:
            self.writer.save_report(self.sample_index, self.counter, report, history)

    def save_iteration_state(self) -> None:
        if not self.writer or self.sample_index is None:
            return
        best_node_id = self.get_best_node_id()
        prediction = self.get_final_prediction(best_node_id)
        self.writer.append_prediction(self.sample_index, self.counter, prediction)


# ============================================================
# 8. Result Writer
# ============================================================


class ResultWriter:
    """集中管理实验输出。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prediction_csv_path(self, index_: int) -> Path:
        return self.output_dir / f"{index_}.csv"

    def report_path(self, index_: int, counter: int) -> Path:
        return self.output_dir / f"INDEX{index_}_{counter}_report.txt"

    def final_result_path(self, index_: int) -> Path:
        return self.output_dir / f"INDEX{index_}_final_result.txt"

    def tree_metadata_path(self, index_: int) -> Path:
        return self.output_dir / f"INDEX{index_}_tree_metadata.json"

    def init_prediction_csv(self, index_: int) -> None:
        with self.prediction_csv_path(index_).open("w", encoding="utf8") as f:
            f.write("iter_times, result\n")

    def append_prediction(self, index_: int, counter: int, prediction: int) -> None:
        with self.prediction_csv_path(index_).open("a", encoding="utf8") as f:
            f.write(f"{counter}, {prediction}\n")

    def save_report(
        self,
        index_: int,
        counter: int,
        report: str,
        history: List[ChatMessage],
    ) -> None:
        with self.report_path(index_, counter).open("w", encoding="utf8") as f:
            f.write("# Generation History\n")
            f.write(json.dumps(history, ensure_ascii=False, indent=2))
            f.write("\n\n# Final Report\n")
            f.write(f"{report}\n")

    def save_final_result(self, index_: int, prediction: int) -> None:
        with self.final_result_path(index_).open("w", encoding="utf8") as f:
            f.write(f"{prediction}\n")

    def save_default_non_vulnerable_rows(self, index_: int, max_iter: int) -> None:
        self.init_prediction_csv(index_)
        for counter in range(max_iter + 1):
            self.append_prediction(index_, counter, 0)
        self.save_final_result(index_, 0)

    def save_tree_metadata(self, index_: int, search_result: SearchResult) -> None:
        payload = {
            "root_id": search_result.root_id,
            "best_node_id": search_result.best_node_id,
            "final_prediction": search_result.final_prediction,
            "error": search_result.error,
            "nodes": [self._node_to_dict(node) for node in search_result.nodes.values()],
        }
        with self.tree_metadata_path(index_).open("w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _usage_to_dict(usage: LLMUsage) -> Dict[str, Any]:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "raw": usage.raw,
        }

    @classmethod
    def _call_record_to_dict(cls, record: LLMCallRecord) -> Dict[str, Any]:
        return {
            "purpose": record.purpose,
            "usage": cls._usage_to_dict(record.usage),
            "elapsed_seconds": record.elapsed_seconds,
            "error": record.error,
        }

    @classmethod
    def _node_to_dict(cls, node: MCTSNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids,
            "depth": node.depth,
            "report": node.report,
            "history": node.history,
            "rewards": node.rewards,
            "q_value": node.q_value,
            "ucb_value": node.ucb_value,
            "visit_count": node.visit_count,
            "generation_usage": cls._usage_to_dict(node.generation_usage),
            "reward_usage": cls._usage_to_dict(node.reward_usage),
            "generation_elapsed_seconds": node.generation_elapsed_seconds,
            "reward_elapsed_seconds": node.reward_elapsed_seconds,
            "generation_call_records": [
                cls._call_record_to_dict(record) for record in node.generation_call_records
            ],
            "reward_call_records": [
                cls._call_record_to_dict(record) for record in node.reward_call_records
            ],
        }


# ============================================================
# 9. Experiment Runner
# ============================================================


class ExperimentRunner:
    """实验运行器。

    负责数据读取、样本循环、搜索对象创建、监督评估和结果汇总。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_client: BaseLLMClient,
        writer: ResultWriter,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.writer = writer
        self.prompt_builder = VulnerabilityPromptBuilder()
        self.parser = VulnerabilityReportParser()
        self.reward_evaluator = SelfCritiqueRewardEvaluator(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            parser=self.parser,
            reward_temperature=self.config.reward_temperature,
            timeout=self.config.timeout,
        )
        self.evaluation_results: List[Dict[str, Any]] = []

    def run(self) -> None:
        data_set = pd.read_csv(self.config.data_csv)

        for index_, data_item in data_set.iterrows():
            if not self.should_process_index(index_):
                continue
            result = self.run_one_example(index_, data_item)
            self.evaluation_results.append(result)

        self.print_overall_accuracy()

    def should_process_index(self, index_: int) -> bool:
        return self.config.index_start <= index_ < self.config.index_end

    def run_one_example(self, index_: int, data_item: pd.Series) -> Dict[str, Any]:
        print(f"\n\n#############################################################")
        print(f"### Index {index_} Analyzing Function ID: {data_item[self.config.id_column]}")
        print(f"#############################################################\n")

        time_start = time.perf_counter()
        function_code = str(data_item[self.config.code_column])
        print(f"func len = {len(function_code)} char.")

        if len(function_code) > self.config.max_function_chars:
            self.writer.save_default_non_vulnerable_rows(index_, self.config.max_iter)
            print(f"### Index {index_} too long. default to non-vul.")
            return self.build_eval_result(
                data_item=data_item,
                final_prediction=0,
                best_report=None,
            )

        search = VulnerabilityMCTSSearch(
            function_code=function_code,
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            parser=self.parser,
            reward_evaluator=self.reward_evaluator,
            writer=self.writer,
            sample_index=index_,
            max_iter=self.config.max_iter,
            max_expand=self.config.max_expand,
            exploration_c=self.config.exploration_c,
            generation_temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

        search_result = search.run()
        final_prediction = search_result.final_prediction
        self.writer.save_final_result(index_, final_prediction)
        self.writer.save_tree_metadata(index_, search_result)

        eval_result = self.build_eval_result(
            data_item=data_item,
            final_prediction=final_prediction,
            best_report=search_result.best_report,
        )

        self.print_example_result(index_, data_item, final_prediction, eval_result, time_start)
        return eval_result

    def build_eval_result(
        self,
        data_item: pd.Series,
        final_prediction: int,
        best_report: Optional[str],
    ) -> Dict[str, Any]:
        ground_truth_label = int(data_item[self.config.label_column])
        is_correct = final_prediction == ground_truth_label
        return {
            "id": data_item[self.config.id_column],
            "ground_truth_label": ground_truth_label,
            "predicted_label": final_prediction,
            "is_correct": is_correct,
            "best_report_found": best_report,
        }

    def print_example_result(
        self,
        index_: int,
        data_item: pd.Series,
        final_prediction: int,
        eval_result: Dict[str, Any],
        time_start: float,
    ) -> None:
        ground_truth_label = int(data_item[self.config.label_column])
        is_correct = eval_result["is_correct"]
        print(f"Ground Truth Vulnerability: {'Yes' if ground_truth_label == 1 else 'No'}")
        print(
            "Final Predicted Vulnerability: "
            f"{'Yes' if final_prediction == 1 else ('No' if final_prediction == 0 else 'Undetermined')}"
        )
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        time_end = time.perf_counter()
        print(
            f"### Index {index_} Total time taken: "
            f"{(time_end - time_start) // 60} min {int((time_end - time_start) % 60)} seconds."
        )
        print("-------------------------------------------------------------")

    def print_overall_accuracy(self) -> None:
        correct_predictions = sum(1 for r in self.evaluation_results if r["is_correct"])
        total_predictions = len(self.evaluation_results)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"\n\n================ OVERALL ACCURACY ================")
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        print(f"==============================================")


# ============================================================
# 10. Main Entry
# ============================================================


def build_llm_client(config: ExperimentConfig) -> BaseLLMClient:
    if config.llm_backend == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        raw_client = LocalTransformersClient(
            model_name_or_path=config.local_model_name_or_path,
            torch_dtype=config.local_torch_dtype,
            device_map=config.local_device_map,
            max_new_tokens=config.max_new_tokens,
            enable_thinking=config.local_enable_thinking,
            think_end_token_id=config.qwen_think_end_token_id,
        )
        cache_identity = "|".join(
            [
                "local",
                config.local_model_name_or_path,
                str(config.max_new_tokens),
                str(config.local_enable_thinking),
            ]
        )
        return maybe_wrap_with_cache(config, raw_client, cache_identity)

    if config.llm_backend == "openai":
        raw_client = OpenAIAPIClient(
            model=config.openai_model,
            api_key_env=config.openai_api_key_env,
            base_url=config.openai_base_url,
            mode=config.openai_mode,
            max_output_tokens=config.max_new_tokens,
        )
        cache_identity = "|".join(
            [
                "openai",
                config.openai_model,
                str(config.openai_base_url),
                config.openai_mode,
                str(config.max_new_tokens),
            ]
        )
        return maybe_wrap_with_cache(config, raw_client, cache_identity)

    raise ValueError(f"Unsupported llm_backend: {config.llm_backend}")


def maybe_wrap_with_cache(
    config: ExperimentConfig,
    raw_client: BaseLLMClient,
    cache_identity: str,
) -> BaseLLMClient:
    if not config.use_llm_cache:
        return raw_client
    return CachedLLMClient(
        inner_client=raw_client,
        cache_dir=config.llm_cache_dir,
        cache_identity=cache_identity,
        enabled=True,
    )


def main() -> None:
    config = ExperimentConfig(
        experiment_name="cwe_119",
        data_csv=PROJECT_ROOT / "data" / "few_test_set_cwe_119.csv",
        llm_backend="openai",
        openai_model=model_name,
        openai_mode="chat",
        openai_base_url=base_url,
        use_llm_cache=True,
        llm_cache_dir=OUTPUTS_DIR / "_llm_cache",
    )

    config.experiment_output_dir.mkdir(parents=True, exist_ok=True)
    llm_client = build_llm_client(config)
    writer = ResultWriter(config.experiment_output_dir)
    runner = ExperimentRunner(config=config, llm_client=llm_client, writer=writer)
    runner.run()


if __name__ == "__main__":
    main()
