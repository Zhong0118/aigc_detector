"""Versioned, gateway-neutral pilot prompts for paired attribution data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    language: str
    domain: str
    task_type: str
    target_length: str
    user_prompt: str
    system_prompt: str | None = None
    version: str = "pilot-v1"
    semantic_group_id: str | None = None
    language_origin: str = "unspecified"
    target_length_spec: str | None = None
    format_type: str = "natural"
    system_prompt_type: str = "none"
    difficulty: str = "core"
    max_output_tokens: int | None = None
    quality_checks: tuple[str, ...] = ()


PILOT_PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        "zh-science-explain-001", "zh", "science", "explanation", "medium",
        "请用约300字向高中生解释为什么天空通常呈蓝色。要求结构清楚，但不要使用项目符号。",
    ),
    PromptSpec(
        "zh-tech-howto-001", "zh", "technology", "procedure", "long",
        "请写一份面向初学者的本地数据备份方案，说明准备、执行、验证和恢复四个阶段，约600字。",
    ),
    PromptSpec(
        "zh-business-compare-001", "zh", "business", "comparison", "medium",
        "一家十人团队需要选择自建服务器或云服务。请比较成本、维护、安全和扩展性，并给出有条件的建议。",
    ),
    PromptSpec(
        "zh-education-argue-001", "zh", "education", "argument", "medium",
        "请论证课堂中应如何合理使用生成式人工智能。需要同时陈述支持理由、风险和折中方案。",
    ),
    PromptSpec(
        "zh-health-summary-001", "zh", "health", "summary", "short",
        "将下面内容压缩成120字以内的摘要：规律睡眠有助于注意力、情绪稳定和记忆巩固。长期睡眠不足会影响判断，并增加学习和工作的错误率。改善睡眠需要固定作息、减少睡前强光和避免过晚摄入咖啡因。",
    ),
    PromptSpec(
        "zh-office-rewrite-001", "zh", "office", "rewrite", "short",
        "把下面的话改写成正式而不过度客套的工作邮件：我们这边数据还没收齐，周五可能交不了，想改到下周二，你看行不行。",
    ),
    PromptSpec(
        "zh-literature-story-001", "zh", "literature", "creative", "medium",
        "写一个约400字的短故事：一名图书管理员在闭馆后发现一本每天都会多出一页的书。结尾保持克制，不要解释全部谜团。",
    ),
    PromptSpec(
        "zh-history-analysis-001", "zh", "history", "analysis", "medium",
        "不引用具体国家，分析印刷技术普及通常会怎样影响知识传播、教育和社会组织。区分短期与长期影响。",
    ),
    PromptSpec(
        "zh-data-interpret-001", "zh", "data_analysis", "data_interpretation", "medium",
        "某服务四周的故障数依次为12、9、15、6，平均修复时间依次为40、35、50、28分钟。请解释趋势、指出不能直接得出的结论，并提出两个后续分析方向。",
    ),
    PromptSpec(
        "zh-reason-debug-001", "zh", "technology", "diagnosis", "medium",
        "一个网页在公司网络中加载缓慢，但手机热点下正常。请列出可能原因，并按排查优先级给出步骤，不要假设已经知道根因。",
    ),
    PromptSpec(
        "zh-policy-notice-001", "zh", "office", "document_writing", "long",
        "为一个研究小组起草约500字的数据文件命名与版本管理规范，包含目的、命名规则、版本号、归档和例外处理。",
    ),
    PromptSpec(
        "zh-general-short-001", "zh", "general", "question_answering", "short",
        "为什么同一个人在不同环境下做决定时可能表现得不一致？请用三段简短文字回答。",
    ),
    PromptSpec(
        "en-science-explain-001", "en", "science", "explanation", "medium",
        "Explain in about 250 words why the sky usually appears blue to a high-school student. Use clear paragraphs and no bullet list.",
    ),
    PromptSpec(
        "en-tech-howto-001", "en", "technology", "procedure", "long",
        "Write a beginner-friendly local data backup plan covering preparation, execution, verification, and recovery in about 500 words.",
    ),
    PromptSpec(
        "en-business-compare-001", "en", "business", "comparison", "medium",
        "A ten-person team must choose between an on-premise server and cloud services. Compare cost, maintenance, security, and scalability, then give a conditional recommendation.",
    ),
    PromptSpec(
        "en-education-argue-001", "en", "education", "argument", "medium",
        "Argue how generative AI should be used responsibly in classrooms. Present supporting reasons, risks, and a practical compromise.",
    ),
    PromptSpec(
        "en-health-summary-001", "en", "health", "summary", "short",
        "Summarize the following in at most 80 words: Regular sleep supports attention, emotional stability, and memory consolidation. Chronic sleep loss impairs judgment and increases mistakes. Helpful practices include a fixed schedule, less bright light before bed, and avoiding caffeine late in the day.",
    ),
    PromptSpec(
        "en-office-rewrite-001", "en", "office", "rewrite", "short",
        "Rewrite this as a professional but direct email: We still do not have all the data, so Friday probably will not work. Could we move the delivery to next Tuesday?",
    ),
    PromptSpec(
        "en-literature-story-001", "en", "literature", "creative", "medium",
        "Write a 350-word short story about a librarian who discovers that a book gains one new page every night after closing. Keep the ending restrained and leave part of the mystery unexplained.",
    ),
    PromptSpec(
        "en-history-analysis-001", "en", "history", "analysis", "medium",
        "Without naming a specific country, analyze how the spread of printing technology can affect knowledge, education, and social organization. Separate short-term from long-term effects.",
    ),
    PromptSpec(
        "en-data-interpret-001", "en", "data_analysis", "data_interpretation", "medium",
        "A service recorded 12, 9, 15, and 6 incidents over four weeks, with mean repair times of 40, 35, 50, and 28 minutes. Explain the trend, identify conclusions the data cannot support, and propose two follow-up analyses.",
    ),
    PromptSpec(
        "en-reason-debug-001", "en", "technology", "diagnosis", "medium",
        "A website is slow on a company network but normal over a phone hotspot. List plausible causes and give a prioritized diagnostic procedure without assuming the root cause.",
    ),
    PromptSpec(
        "en-policy-notice-001", "en", "office", "document_writing", "long",
        "Draft a 450-word file naming and version-control policy for a research group. Include purpose, naming rules, version numbers, archiving, and exception handling.",
    ),
    PromptSpec(
        "en-general-short-001", "en", "general", "question_answering", "short",
        "Why might the same person make inconsistent decisions in different settings? Answer in three short paragraphs.",
    ),
)
