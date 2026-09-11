"""
SiliconFlow API 延迟诊断工具

用途：排查 ainvoke() 51 秒延迟的根因
用法：python tests/test_api_latency.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path（用于导入 config）
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


def load_config():
    """从 config.settings 加载 LLM 配置"""
    from app.core.config.settings import llm_settings
    return {
        "api_key": llm_settings.LLM_API_KEY,
        "base_url": llm_settings.LLM_BASE_URL,
        "model": llm_settings.LLM_MODEL_NAME,
    }


# 模拟意图识别的完整 prompt（3904 字符，与生产环境一致）
INTENT_SYSTEM_TEMPLATE = """你是一个智能意图识别助手。请分析用户输入，完成三项任务。

### 核心原则（最高优先级）
**"宁严勿宽"原则**：如果用户的问题涉及获取具体数据、列表、记录或统计信息，**必须**归类为 `database_query`。只有纯粹的寒暄、问候或与业务完全无关的闲聊才归类为 `daily_chat`。

---

### 任务1：判断意图类型

#### 数据库查询意图 (`database_query`) 的特征
- **获取列表/明细**：询问"全部"、"所有"、"列表"、"名字"、"记录"等（例如："告诉我所有人的名字"、"列出所有设备"）。
- **包含查询条件**：包含具体的人名、部门、时间、设备等筛选条件。
- **统计与聚合**：询问"多少"、"统计"、"总和"、"平均"等。
- **业务实体**：涉及考勤、打卡、用户、部门、设备、订单等业务领域。
- **关键词**：查、找、列出、显示、统计、多少、哪些、全部、所有、名字、记录、信息。

#### 日常聊天意图 (`daily_chat`) 的特征
- **社交礼仪**：你好、再见、谢谢、早上好。
- **能力询问**：你能做什么？你是谁？
- **纯闲聊**：今天天气不错（非查询天气数据）、讲个笑话、心情表达。
- **关于系统/制度的开放性问题**：如"考勤系统好用吗"、"打卡机怎么用"、"考勤数据准不准"、"工资什么时候发"——这些是对系统/制度的主观评价或流程咨询，不是数据查询。
- **注意**：如果用户问"有哪些人"，这是查询，不是聊天。

---

### 任务2：判断是否需要检索长期记忆 (`need_memory`)

`need_memory` 关注的是**这条问题能否独立成立**——是否必须依赖历史对话才能理解。

#### 需要检索 (true) 的情况
- **代词指代**："他"、"她"、"他们"、"那些"、"那个"、"刚才那个"、"上次说的"——指代对象在历史对话中。
- **上下文省略**：问题不完整，必须结合上文（例如："那他们的考勤呢？"、"再给我看一下"）。
- **第一人称归属**："我的"、"我们部门的"——需要从会话/用户上下文确定具体对象。

#### 不需要检索 (false) 的情况
- **独立查询**：问题包含完整的主语和条件（例如："告诉我全部人的名字"、"张三的打卡记录"——人名是常驻实体，不属于代词指代）。
- **通用闲聊**：问候语、能力询问。

**重要**：`need_memory` 只看"指代/省略"，不看"复杂度"。"对比研发部和市场部"虽然复杂但不依赖历史，应为 false。

---

### 任务3：判断是否需要多步推理 (`needs_react`)

`needs_react` 关注的是**这条查询能否用一条 SQL 直接搞定**——还是需要拆解成多步、多次查询、互相依赖。

#### 需要多步推理 (true) 的情况（仅在 intent=database_query 时考虑）
- **多对象对比**：含"对比"、"vs"、"和...哪个"、"差异"，需要分别查再对比。
- **趋势/异常分析**：含"趋势"、"变化"、"异常"、"连续"、"分布"，需要先找规律再筛选。
- **嵌套筛选**：条件之间互相依赖（"既...又"、"在A中且不在B中"、"打卡时间和上班时间相差超过30分钟"）。
- **多步聚合**：需要先聚合再聚合（"按周统计每个部门的平均工作时长"、"出勤率排名"）。
- **跨实体关联**：需要 3 张及以上表关联且含子查询（"跨部门协作项目中各成员的考勤"）。

#### 不需要多步推理 (false) 的情况
- 单表查询：所有用户、设备列表、张三的打卡记录。
- 简单聚合：总人数、研发部多少人、今天迟到几人。
- 简单两表 JOIN：李四在哪个部门、每个部门的平均工资。
- 所有 `daily_chat` 意图——`needs_react` 必须为 false。

---

### Few-shot 示例（重点关注边界场景）

#### 示例1：业务词 ≠ 查询（这些是闲聊）
- 输入：`考勤系统好用吗`
  → `{{intent: daily_chat, need_memory: false, needs_react: false}}`
  理由：对系统的主观评价，不需要查任何数据。
- 输入：`工资什么时候发`
  → `{{intent: daily_chat, need_memory: false, needs_react: false}}`
  理由：流程咨询，不是数据查询。
- 输入：`考勤数据准不准`
  → `{{intent: daily_chat, need_memory: false, needs_react: false}}`
  理由：对系统准确性的疑问，不是要获取具体数据。

#### 示例2：口语化查询（这些是 query，不要被语气骗）
- 输入：`今天出勤怎么样`
  → `{{intent: database_query, need_memory: false, needs_react: false}}`
  理由：要"今天的出勤数据"，是查询不是闲聊。
- 输入：`有没有什么异常`
  → `{{intent: database_query, need_memory: false, needs_react: false}}`
  理由：在业务上下文中"异常"指考勤异常记录，是查询。
- 输入：`最近加班多不多`
  → `{{intent: database_query, need_memory: false, needs_react: false}}`
  理由：要加班数据的统计判断。

#### 示例3：需要记忆 vs 不需要记忆
- 输入：`张三的打卡记录`
  → `{{need_memory: false}}`
  理由：人名是独立实体，不是代词指代。
- 输入：`我的考勤怎么样`
  → `{{need_memory: true}}`
  理由："我的"是第一人称归属，需从会话上下文确定 user_id。
- 输入：`我们部门的打卡数据`
  → `{{need_memory: true}}`
  理由："我们部门"需要从用户上下文确定具体部门。
- 输入：`那他们的考勤呢`
  → `{{need_memory: true, needs_react: false}}`
  理由：代词指代要查记忆；但本身只是一条简单查询，不需要 ReAct。

#### 示例4：需要 ReAct vs 不需要 ReAct
- 输入：`对比研发部和市场部的考勤情况`
  → `{{needs_react: true}}`
  理由：两个对象对比，需要分别查再比较。
- 输入：`分析各部门每月的加班趋势`
  → `{{needs_react: true}}`
  理由：含"趋势"，需要多步聚合 + 时间维度分析。
- 输入：`查询既在A部门又在B部门兼职的员工`
  → `{{needs_react: true}}`
  理由：嵌套筛选条件互相依赖。
- 输入：`查询打卡时间与上班时间相差超过30分钟的异常记录`
  → `{{needs_react: true}}`
  理由：需要先计算差值再筛选，单条 SQL 写起来也复杂、需要先理解 schema。
- 输入：`显示每个部门的平均工资`
  → `{{needs_react: false}}`
  理由：一条 GROUP BY 即可，不需要拆步。
- 输入：`今天有多少人迟到`
  → `{{needs_react: false}}`
  理由：单条聚合查询。
- 输入：`查询打卡次数最多的前10名员工`
  → `{{needs_react: false}}`
  理由：ORDER BY + LIMIT 一条 SQL 搞定。

---

请按以下 JSON 格式返回（所有字段必须填写）：
{{
  "intent": "database_query 或 daily_chat",
  "reasoning": "意图判断理由",
  "need_memory": true 或 false,
  "memory_reason": "是否需要检索记忆的理由（聚焦指代/省略，不要混入复杂度）",
  "needs_react": true 或 false,
  "react_reason": "是否需要多步推理的理由（聚焦能否一条SQL搞定）
}}"""


async def test_single_request(client, config, test_input, round_num):
    """执行单次 API 调用并记录详细时间节点"""

    print(f"\n{'='*60}")
    print(f"🔄 第 {round_num} 轮测试")
    print(f"📝 测试输入: {test_input[:50]}...")
    print(f"{'='*60}")

    t_start = time.perf_counter()

    # 构造请求 payload（模拟生产环境的意图识别请求）
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_TEMPLATE},
            {"role": "user", "content": f"用户输入：{test_input}"}
        ],
        "temperature": 0,
        "max_tokens": 500,
        # 启用 structured output（function calling 模式）
        "tools": [{
            "type": "function",
            "function": {
                "name": "IntentResult",
                "description": "意图识别结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string", "enum": ["database_query", "daily_chat"]},
                        "reasoning": {"type": "string"},
                        "need_memory": {"type": "boolean"},
                        "memory_reason": {"type": "string"},
                        "needs_react": {"type": "boolean"},
                        "react_reason": {"type": "string"}
                    },
                    "required": ["intent", "reasoning", "need_memory", "memory_reason", "needs_react", "react_reason"]
                }
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "IntentResult"}}
    }

    t_payload_ready = time.perf_counter()
    print(f"[{(t_payload_ready - t_start)*1000:8.0f}ms] ✅ Payload 构建完成 ({len(json.dumps(payload))} 字节)")

    try:
        # 发送 HTTP 请求
        t_before_send = time.perf_counter()
        response = await client.post(
            f"{config['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        t_after_response = time.perf_counter()

        print(f"[{(t_after_response - t_start)*1000:8.0f}ms] 📡 HTTP 响应接收 (状态码: {response.status_code})")

        if response.status_code == 200:
            result = response.json()
            t_parsed = time.perf_counter()
            print(f"[{(t_parsed - t_start)*1000:8.0f}ms] 📊 JSON 解析完成")

            # 提取关键指标
            usage = result.get("usage", {})
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            tool_calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

            print(f"\n📈 Token 使用:")
            print(f"   Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens:      {usage.get('total_tokens', 'N/A')}")

            print(f"\n⏱️ 时间分解:")
            print(f"   Payload 构建:      {(t_payload_ready - t_start)*1000:.0f}ms")
            print(f"   HTTP 往返(TTFB):   {(t_after_response - t_before_send)*1000:.0f}ms")  # Time To First Byte
            print(f"   JSON 解析:         {(t_parsed - t_after_response)*1000:.0f}ms")
            print(f"   总耗时:            {(t_parsed - t_start)*1000:.0f}ms")

            if tool_calls:
                print(f"\n✅ 结构化输出成功 (tool_calls):")
                for tc in tool_calls:
                    func_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    print(f"   intent:       {func_args.get('intent')}")
                    print(f"   need_memory:  {func_args.get('need_memory')}")
                    print(f"   needs_react:  {func_args.get('needs_react')}")
            else:
                print(f"\n⚠️ 未返回 tool_calls (原始内容前100字): {content[:100]}")

            return {
                "round": round_num,
                "total_ms": (t_parsed - t_start) * 1000,
                "http_ms": (t_after_response - t_before_send) * 1000,
                "status_code": response.status_code,
                "success": True
            }

        else:
            print(f"\n❌ HTTP 错误: {response.status_code}")
            print(f"   响应内容: {response.text[:200]}")
            return {
                "round": round_num,
                "total_ms": (time.perf_counter() - t_start) * 1000,
                "http_ms": 0,
                "status_code": response.status_code,
                "success": False,
                "error": response.text[:200]
            }

    except Exception as e:
        t_error = time.perf_counter()
        print(f"\n❌ 异常: {str(e)}")
        print(f"   总耗时: {(t_error - t_start)*1000:.0f}ms")
        return {
            "round": round_num,
            "total_ms": (t_error - t_start) * 1000,
            "http_ms": 0,
            "status_code": 0,
            "success": False,
            "error": str(e)
        }


async def run_latency_test():
    """运行完整的延迟测试套件"""

    print("=" * 70)
    print("🔬 SiliconFlow API 延迟诊断工具")
    print("=" * 70)

    # 加载配置
    config = load_config()
    print(f"\n📋 配置信息:")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Model:     {config['model']}")
    print(f"   API Key:   {config['api_key'][:20]}...{config['api_key'][-4:]}")

    # 创建 httpx 客户端（复用连接池，模拟生产环境）
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
    ) as client:

        print("\n✅ HTTP 客户端已创建（连接池模式）")

        # 测试用例列表
        test_cases = [
            "先查一下哪些人没打卡，然后统计按部门分组的缺勤人数，再和上月对比",  # 复杂问题（你遇到的慢场景）
            "今天有多少人迟到",  # 简单问题
            "你好",  # 最简单的问题
        ]

        results = []

        for i, test_input in enumerate(test_cases, 1):
            result = await test_single_request(client, config, test_input, i)
            results.append(result)

            # 如果不是最后一轮，等待 2 秒避免限流
            if i < len(test_cases):
                print(f"\n⏳ 等待 2 秒后进行下一轮测试...")
                await asyncio.sleep(2)

        # 输出汇总报告
        print("\n" + "=" * 70)
        print("📊 测试汇总报告")
        print("=" * 70)

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            avg_total = sum(r["total_ms"] for r in successful) / len(successful)
            avg_http = sum(r["http_ms"] for r in successful) / len(successful)

            print(f"\n✅ 成功请求: {len(successful)}/{len(results)}")
            print(f"   平均总耗时:  {avg_total:.0f}ms ({avg_total/1000:.1f}s)")
            print(f"   平均HTTP耗时: {avg_http:.0f}ms ({avg_http/1000:.1f}s)")
            print(f"   平均非网络开销: {avg_total - avg_http:.0f}ms ({(avg_total - avg_http)/1000:.1f}s)")

            print(f"\n📋 各轮详情:")
            for r in successful:
                status = "✅" if r["success"] else "❌"
                print(f"   第 {r['round']} 轮: {status} 总耗时={r['total_ms']:.0f}ms, HTTP={r['http_ms']:.0f}ms")

        if failed:
            print(f"\n❌ 失败请求: {len(failed)}")
            for r in failed:
                print(f"   第 {r['round']} 轮: {r.get('error', '未知错误')}")

        # 给出诊断建议
        print("\n💡 诊断建议:")
        if successful:
            if avg_http > 30000:  # HTTP 超过 30 秒
                print("   🔴 [严重] HTTP 往返时间过长 (>30s)")
                print("   可能原因:")
                print("   1. SiliconFlow 服务端排队/限流")
                print("   2. 网络延迟（DNS/TCP/TLS）")
                print("   3. API 提供商服务器负载过高")
                print("")
                print("   建议:")
                print("   - 检查 SiliconFlow 控制台的用量和配额")
                print("   - 尝试更换网络环境或使用代理")
                print("   - 联系 SiliconFlow 技术支持确认服务状态")
            elif avg_http > 10000:  # HTTP 超过 10 秒
                print("   🟡 [警告] HTTP 往返时间较长 (10-30s)")
                print("   可能是模型推理时间较长或中等程度的网络延迟")
            elif avg_http > 5000:  # HTTP 超过 5 秒
                print("   🟢 [正常] HTTP 往返时间可接受 (5-10s)")
                print("   复杂 prompt 的正常推理时间")
            else:
                print("   🟢 [优秀] HTTP 往返时间很快 (<5s)")

            non_network_overhead = avg_total - avg_http
            if non_network_overhead > 5000:  # 非网络开销超过 5 秒
                print(f"\n   🔴 [异常] 非网络开销过大: {non_network_overhead:.0f}ms")
                print("   这说明 LangChain 内部有性能问题")
                print("   可能原因:")
                print("   - with_structured_output() 初始化缓慢")
                print("   - Pydantic model validation 开销大")
                print("   - LangChain 内部的序列化/反序列化")


if __name__ == "__main__":
    asyncio.run(run_latency_test())
