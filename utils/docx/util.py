import re
from docx import Document
from typing import Tuple, List



def slice_document(file_path: str) -> Tuple[List[str], List[dict]]:
    """读取并切片文档，返回纯文本列表和元数据列表"""
    try:
        doc = Document(file_path)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'")
        return [], []

    print(f"✅ 文件已打开，开始切片...")

    chunks_content = []
    chunks_meta = []

    current_chunk_text = []
    current_table_name = "未知表"

    # 🔧 核心修复：正则只匹配英文、数字、下划线
    # 解释：[a-zA-Z0-9_]+ 只会匹配类似 sys_user, onl_cgform_head 这样的纯英文表名
    # 遇到中文括号 '（' 或空格就会停止匹配，彻底杜绝超长问题
    title_pattern = re.compile(r"^\d+\.\s*表名：\s*([a-zA-Z0-9_]+)")

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        match = title_pattern.search(text)

        if match:
            # 1. 保存上一个切片
            if current_chunk_text:
                full_text = "\n".join(current_chunk_text)
                chunks_content.append(full_text)

                # 防御性截断（虽然新正则应该不需要了，但保留以防万一）
                safe_name = current_table_name[:2000]
                chunks_meta.append({"table_name": safe_name})

                display_name = safe_name if len(safe_name) <= 20 else safe_name[:20] + "..."
                print(f"✂️ 已保存: [{display_name}] (内容长度: {len(full_text)} 字)")

            # 2. 开启新的切片
            current_table_name = match.group(1)  # 这里现在只会拿到 'sys_user'
            current_chunk_text = [text]

            # 🔍 调试打印：确认拿到的表名很短
            print(f"🔎 匹配到新表名: '{current_table_name}' (长度: {len(current_table_name)})")

        else:
            if current_chunk_text:
                current_chunk_text.append(text)

    # 3. 保存最后一个切片
    if current_chunk_text:
        full_text = "\n".join(current_chunk_text)
        chunks_content.append(full_text)

        safe_name = current_table_name[:2000]
        chunks_meta.append({"table_name": safe_name})

        display_name = safe_name if len(safe_name) <= 20 else safe_name[:20] + "..."
        print(f"✂️ 已保存: [{display_name}] (内容长度: {len(full_text)} 字)")

    print(f"🎉 切片完成！共 {len(chunks_content)} 块文本，{len(chunks_meta)} 条元数据。")

    if len(chunks_content) != len(chunks_meta):
        print(f"❌ 内部错误：文本数与元数据数不一致！")
        return [], []

    return chunks_content, chunks_meta

