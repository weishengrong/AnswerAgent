"""查看 ChromaDB 向量库里的所有具体内容"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config.settings import rag_settings
import chromadb

client = chromadb.PersistentClient(path=rag_settings.RAG_CHROMA_PERSIST_DIR)
col = client.get_collection("try")

all_data = col.get(limit=500)

print(f"总文档数: {len(all_data['documents'])}\n")

for i, (meta, doc) in enumerate(zip(all_data["metadatas"], all_data["documents"])):
    ct = meta.get("chunk_type", "?")
    tbl = meta.get("table_name", "?")
    col_name = meta.get("column_name", "")
    sep = "=" * 60
    print(sep)
    print(f"[{i+1}] chunk_type={ct} | table={tbl}", end="")
    if col_name:
        print(f" | column={col_name}", end="")
    print()
    print(sep)
    print(doc)
    print()