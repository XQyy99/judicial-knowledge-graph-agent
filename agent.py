import os
import json
import fnmatch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv()

client = OpenAI()

DEFAULT_IGNORE_PATTERNS = [
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "target",
    "vendor",
]

DEFAULT_INCLUDE_EXTS = [
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".kt", ".rb",
    ".php", ".rs", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".m",
    ".json", ".yml", ".yaml", ".toml", ".md"
]

MAX_FILE_CHARS = 12000
MAX_TOTAL_CHARS = 45000

def should_ignore(path: Path, ignore_patterns: List[str]) -> bool:
    parts = set(path.parts)
    for pat in ignore_patterns:
        if pat in parts:
            return True
    return False

def is_supported_file(path: Path, include_exts: List[str]) -> bool:
    return path.suffix.lower() in include_exts

def read_text_file(path: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception:
            return ""
    except Exception:
        return ""

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[TRUNCATED]..."
    return text

def collect_files(
    repo_path: Path,
    ignore_patterns: List[str],
    include_exts: List[str],
) -> List[Path]:
    files = []
    for p in repo_path.rglob("*"):
        if p.is_file() and not should_ignore(p, ignore_patterns) and is_supported_file(p, include_exts):
            files.append(p)
    return files

def rank_files(files: List[Path]) -> List[Path]:
    """
    简单排序策略：
    1. 优先核心文件：README、main、app、index、server、api、service、controller、router
    2. 其次短文件
    """
    keywords = [
        "readme", "main", "app", "index", "server", "api",
        "service", "controller", "router", "handler", "model", "util"
    ]

    def score(path: Path) -> Tuple[int, int]:
        name = path.name.lower()
        priority = 0
        for i, kw in enumerate(keywords):
            if kw in name:
                priority += (len(keywords) - i) * 10
        # 越短越优先
        size_score = -len(str(path))
        return (priority, size_score)

    return sorted(files, key=score, reverse=True)

def build_repo_context(repo_path: Path, files: List[Path]) -> Dict[str, str]:
    total = 0
    context = {}

    for f in tqdm(files, desc="Reading files"):
        text = read_text_file(f)
        if not text:
            continue

        if total + len(text) > MAX_TOTAL_CHARS:
            remain = MAX_TOTAL_CHARS - total
            if remain <= 0:
                break
            text = text[:remain] + "\n\n...[TRUNCATED TOTAL CONTEXT]..."
            context[str(f.relative_to(repo_path))] = text
            break

        context[str(f.relative_to(repo_path))] = text
        total += len(text)

    return context

def create_review_prompt(repo_name: str, context: Dict[str, str], user_focus: str = "") -> str:
    files_block = []
    for file_path, content in context.items():
        files_block.append(f"## FILE: {file_path}\n```text\n{content}\n```")

    focus_text = f"\n用户特别关注点：{user_focus}\n" if user_focus else ""

    return f"""
你是一个资深的代码审查与重构 Agent，目标是帮助这个仓库提升：
1. 可读性
2. 可维护性
3. 安全性
4. 性能
5. 架构清晰度

请基于下面的仓库代码内容，输出一份“可执行”的审查结果。

仓库名：{repo_name}
{focus_text}

请严格按以下 JSON 格式输出，不要输出额外文本：

{{
  "summary": "一句话总结整体问题",
  "overall_score": 0-100,
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "title": "问题标题",
      "file": "文件路径",
      "line_hint": "尽量给出行号或位置提示，没有就写 null",
      "problem": "问题描述",
      "why_it_matters": "为什么重要",
      "suggested_fix": "建议修改方式",
      "refactor_example": "可选：给出简短代码片段或伪代码"
    }}
  ],
  "refactor_plan": [
    "按优先级排序的重构计划 1",
    "按优先级排序的重构计划 2"
  ],
  "quick_wins": [
    "能快速落地的小优化 1",
    "能快速落地的小优化 2"
  ],
  "risks": [
    "如果不改可能带来的风险 1",
    "如果不改可能带来的风险 2"
  ]
}}

仓库内容如下：

{chr(10).join(files_block)}
""".strip()

def call_llm_review(prompt: str, model: str) -> Dict:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )

    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试从输出中提取 JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def create_markdown_report(result: Dict) -> str:
    findings = result.get("findings", [])
    refactor_plan = result.get("refactor_plan", [])
    quick_wins = result.get("quick_wins", [])
    risks = result.get("risks", [])

    md = []
    md.append(f"# 代码审查报告\n")
    md.append(f"**总分：** {result.get('overall_score', 'N/A')}/100\n")
    md.append(f"**概述：** {result.get('summary', '')}\n")

    md.append("## 问题清单\n")
    if findings:
        for i, item in enumerate(findings, 1):
            md.append(f"### {i}. [{item.get('severity', '').upper()}] {item.get('title', '')}")
            md.append(f"- **文件**：`{item.get('file', '')}`")
            md.append(f"- **位置**：{item.get('line_hint', 'N/A')}")
            md.append(f"- **问题**：{item.get('problem', '')}")
            md.append(f"- **影响**：{item.get('why_it_matters', '')}")
            md.append(f"- **建议**：{item.get('suggested_fix', '')}")
            if item.get("refactor_example"):
                md.append(f"- **示例**：\n```text\n{item.get('refactor_example')}\n```")
            md.append("")
    else:
        md.append("没有发现明显问题。\n")

    md.append("## 重构计划\n")
    for item in refactor_plan:
        md.append(f"- {item}")
    md.append("")

    md.append("## 快速收益\n")
    for item in quick_wins:
        md.append(f"- {item}")
    md.append("")

    md.append("## 风险\n")
    for item in risks:
        md.append(f"- {item}")
    md.append("")

    return "\n".join(md)

def save_outputs(output_dir: Path, result: Dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = create_markdown_report(result)
    (output_dir / "review_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "review_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def generate_patch_prompt(context: Dict[str, str], review_result: Dict) -> str:
    """
    让模型根据审查结果，生成统一 diff 风格补丁建议。
    注意：这是“草案补丁”，不保证 100% 可直接应用。
    """
    findings = review_result.get("findings", [])
    files_block = []
    for file_path, content in context.items():
        files_block.append(f"## FILE: {file_path}\n```text\n{content}\n```")

    return f"""
你是一个高级重构工程师。请根据下面的代码与审查结果，生成尽量可应用的统一 diff 补丁建议。

要求：
1. 优先修复高严重度问题
2. 尽量少改动，保持风格一致
3. 如果无法给出完整补丁，请给出针对关键文件的局部 diff
4. 只输出 unified diff，不要解释，不要输出额外文本

审查结果：
{json.dumps(review_result, ensure_ascii=False, indent=2)}

仓库代码：
{chr(10).join(files_block)}
""".strip()

def call_llm_patch(prompt: str, model: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Code Review / Refactor Agent")
    parser.add_argument("repo", help="仓库路径")
    parser.add_argument("--output", default="./agent_output", help="输出目录")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1"), help="模型名")
    parser.add_argument("--focus", default="", help="审查重点，例如：安全/性能/架构/前端")
    parser.add_argument("--patch", action="store_true", help="是否生成 patch.diff")
    parser.add_argument("--max-files", type=int, default=30, help="最多读取多少个文件")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    output_dir = Path(args.output).resolve()

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"仓库路径不存在或不是目录: {repo_path}")

    files = collect_files(repo_path, DEFAULT_IGNORE_PATTERNS, DEFAULT_INCLUDE_EXTS)
    files = rank_files(files)[:args.max_files]

    if not files:
        raise RuntimeError("没有找到可分析的文件。")

    context = build_repo_context(repo_path, files)

    prompt = create_review_prompt(
        repo_name=repo_path.name,
        context=context,
        user_focus=args.focus
    )

    print("开始代码审查...")
    review_result = call_llm_review(prompt, args.model)

    save_outputs(output_dir, review_result)
    print(f"审查报告已保存到: {output_dir / 'review_report.md'}")

    if args.patch:
        print("开始生成补丁建议...")
        patch_prompt = generate_patch_prompt(context, review_result)
        patch_text = call_llm_patch(patch_prompt, args.model)
        (output_dir / "patch.diff").write_text(patch_text, encoding="utf-8")
        print(f"补丁已保存到: {output_dir / 'patch.diff'}")

if __name__ == "__main__":
    main()