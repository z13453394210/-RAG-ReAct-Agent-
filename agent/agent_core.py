"""
rag-knowledge-agent — Agent 核心

使用 LangChain AgentExecutor 编排 LLM + 工具。
采用 ReAct 模式（Thought/Action/Observation），兼容 DeepSeek 等非 OpenAI 模型。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agent.tools import ALL_TOOLS, get_tools

# 从项目根目录加载 .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))

# ---------------------------------------------------------------------------
# ReAct Prompt
# ---------------------------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template("""你是知识库问答助手，使用以下工具来回答问题：

{tools}

工具名称: {tool_names}

回答规则：
- 关于文档的问题，调用 knowledge_base 工具搜索
- 如果文档是英文，用英文关键词搜索（如 abstract、method、title 等）
- 如果第一次搜索不理想，换不同的关键词多搜几次
- 总结文档时把所有找到的信息整合起来回答
- 引用知识库内容时注明来源
- 找不到时如实告知
- 用中文回答

按以下格式输出：

Question: 用户的问题
Thought: 思考下一步该做什么
Action: 工具名称（必须从上面的列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
...（根据需要重复 Thought/Action/Observation）
Thought: 我现在可以回答
Final Answer: 最终答案

用户的问题是：
{input}

{agent_scratchpad}""")


# ---------------------------------------------------------------------------
# Agent 工厂
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def create_agent(
    model_name: str = "deepseek-chat",
    temperature: float = 0.0,
    verbose: bool = True,
    tool_names: Optional[list[str]] = None,
) -> AgentExecutor:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "sk-your-deepseek-api-key-here":
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置。请在 .env 文件中配置。"
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    tools = get_tools(tool_names)
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )