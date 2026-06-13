"""
rag-knowledge-agent — Streamlit 前端

提供:
  - 侧边栏：文档上传、查看已索引文档、清空向量库
  - 主面板：与多工具 Agent 的聊天界面（DeepSeek 驱动）
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.rag import ingest_document, list_indexed_sources, delete_all_documents
from agent.agent_core import create_agent
from langchain.schema import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG 知识库问答助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

# 隐藏 Streamlit 自带的英文菜单
st.markdown("""<style>
/* 隐藏右上角英文菜单按钮 */
button[data-testid="baseButton-header"] { display: none !important; }
/* 隐藏 "Made with Streamlit" 页脚 */
footer { display: none !important; }
/* 隐藏部署按钮 */
[data-testid="stAppDeployButton"] { display: none !important; }
/* 隐藏设置、打印等其他菜单项 */
div[data-testid="stHeaderToolbar"] button:not([data-testid="baseButton-header"]) { display: none !important; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 侧边栏 — 文档管理
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 RAG 知识库助手")
    st.markdown("上传文档后，用自然语言提问即可。")
    st.divider()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "sk-your-deepseek-api-key-here":
        st.error("⚠️ 未配置 DeepSeek API Key\n\n请在项目根目录创建 `.env` 文件，填入:\n\n`DEEPSEEK_API_KEY=sk-...`\n\n注册: https://platform.deepseek.com")
        st.stop()

    model_name = st.selectbox("DeepSeek 模型", options=["deepseek-chat", "deepseek-reasoner"], index=0, help="deepseek-chat = V3（快速通用）; deepseek-reasoner = R1（深度推理）")
    st.divider()

    st.subheader("📄 上传文档")
    uploaded_file = st.file_uploader("选择文件", type=["pdf", "docx", "txt", "md"], label_visibility="collapsed")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
        upload_dir = PROJECT_ROOT / "docs" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        with st.spinner(f"正在索引 {uploaded_file.name}..."):
            file_bytes = uploaded_file.getvalue()
            file_path.write_bytes(file_bytes)
            try:
                num_chunks = ingest_document(str(file_path))
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ **{uploaded_file.name}** 索引完成（共 {num_chunks} 个文本块）")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 索引失败: {e}")
                st.session_state.processed_files.discard(uploaded_file.name)

    st.subheader("📚 知识库")
    sources = list_indexed_sources()
    if sources:
        st.caption(f"已索引 {len(sources)} 篇文档:")
        for s in sources:
            st.markdown(f"- {s}")
    else:
        st.caption("暂无已索引的文档，请上传文件。")

    if sources and st.button("🗑️ 清空所有文档", type="secondary", use_container_width=True):
        deleted = delete_all_documents()
        st.success(f"已删除 {deleted} 条记录，重新上传即可重建索引。")
        st.rerun()

    st.divider()
    with st.expander("💡 试试这样问"):
        st.markdown("- *总结一下上传的报告*\n- *上海的天气怎么样？*\n- *搜一下最新的 AI 新闻*\n- *计算 3400 的 15%*\n- *文档里关于 XX 怎么说？*")

# ---------------------------------------------------------------------------
# 主面板 — 聊天界面
# ---------------------------------------------------------------------------
st.title("💬 与知识库对话")
st.caption("Agent 由 DeepSeek 驱动，可搜索文档、浏览网页、查询天气、做数学计算。")

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="你好！我是你的 RAG 知识库助手。\n\n在左侧侧边栏上传文档，然后向我提问即可。")]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = create_agent(model_name=model_name, verbose=False)

    chat_history = []
    for m in st.session_state.messages[:-1]:
        if isinstance(m, HumanMessage):
            chat_history.append(("human", m.content))
        elif isinstance(m, AIMessage):
            chat_history.append(("ai", m.content))

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = agent.invoke({"input": prompt, "chat_history": chat_history})
                response = result["output"]
            except Exception as e:
                response = f"抱歉，遇到了错误: {e}"
        st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))
    st.rerun()