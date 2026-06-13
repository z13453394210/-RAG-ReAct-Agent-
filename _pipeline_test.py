import sys, os, traceback
sys.path.insert(0, ".")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    from agent.rag import ingest_document, list_indexed_sources
    os.makedirs("docs/uploads", exist_ok=True)
    with open("docs/uploads/test.txt", "w", encoding="utf-8") as f:
        f.write("公司政策：每年有15天带薪年假。远程办公需要经理批准。")
    n = ingest_document("docs/uploads/test.txt")
    src = list_indexed_sources()
    with open("_result.txt", "w") as f:
        f.write(f"OK: chunks={n} sources={src}")
except Exception as e:
    with open("_error.txt", "w") as f:
        f.write(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(file=f)
