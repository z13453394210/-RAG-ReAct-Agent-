# ============================================
# RAG Knowledge Agent — 示例文档灌入脚本
# ============================================
# 运行: python scripts/seed_data.py
# 会向知识库中灌入示例文档，方便快速测试。

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.rag import ingest_document

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "docs" / "uploads"

SAMPLE_TEXTS = {
    "公司政策.txt": """公司员工手册

1. 工作时间
标准工作时间为周一至周五上午 9:00 至下午 6:00，
中午 1 小时午休。经经理批准可灵活安排工作时间。

2. 远程办公
员工每周最多可远程办公 3 天。完全远程工作需要副总裁级别批准。

3. 年假政策
正式员工每年享有 15 天带薪年假。未休完的年假最多可结转 5 天至下一年。

4. 行为准则
所有员工必须完成年度道德培训。严禁任何形式的骚扰行为。
""",

    "产品介绍.md": "# 产品介绍\n\n"
    "我们的旗舰产品 **DataSync Pro** 是一款实时数据集成平台，"
    "专为企业团队设计。\n\n"
    "## 主要功能\n\n"
    "- **实时同步**：源与目标之间亚秒级延迟\n"
    "- **200+ 连接器**：预置 Salesforce、HubSpot、Snowflake、BigQuery 等集成\n"
    "- **数据转换**：可视化 ETL 流水线构建器，支持 Python 脚本\n"
    "- **监控告警**：内置数据质量与流水线健康仪表板\n\n"
    "## 定价\n\n"
    "| 套餐 | 价格 | 月事件量 |\n"
    "|------|------|---------|\n"
    "| 入门版 | ¥99 | 100 万 |\n"
    "| 专业版 | ¥499 | 1000 万 |\n"
    "| 企业版 | 定制 | 无限 |\n",
}


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for fname, content in SAMPLE_TEXTS.items():
        path = SAMPLE_DIR / fname
        path.write_text(content, encoding="utf-8")
        n_chunks = ingest_document(str(path))
        print(f"  ✅ {fname} → {n_chunks} 个文本块")
    print("\n灌入完成！现在可以询问关于公司政策或产品信息的问题。")


if __name__ == "__main__":
    main()