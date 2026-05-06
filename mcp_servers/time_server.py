from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json

mcp = FastMCP("time-tools")


@mcp.tool()
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前精确时间，支持指定时区。返回格式：YYYY-MM-DD HH:MM:SS"""
    import pytz
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def get_date_range(period: str) -> str:
    """将自然语言时间范围转为精确的起止日期。

    支持的 period 值：
    - 今天、昨天、前天
    - 本周、上周
    - 本月、上月、上上月
    - 今年、去年
    - 近7天、近30天、近90天

    返回 JSON: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "description": "..."}
    """
    today = datetime.now().date()
    start = end = today
    desc = ""

    period_lower = period.strip().lower()

    if period_lower == "今天":
        start = end = today
        desc = "今天"
    elif period_lower == "昨天":
        start = end = today - timedelta(days=1)
        desc = "昨天"
    elif period_lower == "前天":
        start = end = today - timedelta(days=2)
        desc = "前天"
    elif period_lower == "本周":
        start = today - timedelta(days=today.weekday())
        end = today
        desc = "本周（周一到今天）"
    elif period_lower == "上周":
        start_of_this_week = today - timedelta(days=today.weekday())
        start = start_of_this_week - timedelta(days=7)
        end = start_of_this_week - timedelta(days=1)
        desc = "上周（周一到周日）"
    elif period_lower == "本月":
        start = today.replace(day=1)
        end = today
        desc = "本月（1号到今天）"
    elif period_lower in ("上月", "上个月"):
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)
        desc = f"上月（{start} 到 {end}）"
    elif period_lower in ("上上月", "上上个月"):
        first_of_this_month = today.replace(day=1)
        first_of_last_month = (first_of_this_month - timedelta(days=1)).replace(day=1)
        end = first_of_last_month - timedelta(days=1)
        start = end.replace(day=1)
        desc = f"上上月（{start} 到 {end}）"
    elif period_lower == "今年":
        start = today.replace(month=1, day=1)
        end = today
        desc = f"今年（{start} 到 {end}）"
    elif period_lower == "去年":
        start = today.replace(year=today.year - 1, month=1, day=1)
        end = today.replace(year=today.year - 1, month=12, day=31)
        desc = f"去年（{start} 到 {end}）"
    elif period_lower.startswith("近") and period_lower.endswith("天"):
        try:
            days = int(period_lower[1:-1])
            start = today - timedelta(days=days - 1)
            end = today
            desc = f"近{days}天（{start} 到 {end}）"
        except ValueError:
            return json.dumps({"error": f"无法解析时间范围：{period}"}, ensure_ascii=False)
    else:
        return json.dumps({"error": f"不支持的时间范围：{period}，支持：今天/昨天/前天/本周/上周/本月/上月/上上月/今年/去年/近N天"}, ensure_ascii=False)

    return json.dumps({
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "description": desc
    }, ensure_ascii=False)


@mcp.tool()
def get_relative_date(expression: str) -> str:
    """将自然语言相对日期表达式转为精确日期。

    支持的表达式：
    - N天前、N天后（如 3天前、7天后）
    - N周前、N周后
    - N月前、N月后
    - 上周一、下周五 等（上周/下周+星期几）
    - 月初、月末、上月初、上月末

    返回 JSON: {"date": "YYYY-MM-DD", "expression": "..."}
    """
    today = datetime.now().date()
    result_date = today
    expr = expression.strip()

    if expr.endswith("天前"):
        try:
            n = int(expr.replace("天前", ""))
            result_date = today - timedelta(days=n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.endswith("天后"):
        try:
            n = int(expr.replace("天后", ""))
            result_date = today + timedelta(days=n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.endswith("周前"):
        try:
            n = int(expr.replace("周前", ""))
            result_date = today - timedelta(weeks=n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.endswith("周后"):
        try:
            n = int(expr.replace("周后", ""))
            result_date = today + timedelta(weeks=n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.endswith("月前"):
        try:
            n = int(expr.replace("月前", ""))
            result_date = today + relativedelta(months=-n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.endswith("月后"):
        try:
            n = int(expr.replace("月后", ""))
            result_date = today + relativedelta(months=n)
        except ValueError:
            return json.dumps({"error": f"无法解析：{expression}"}, ensure_ascii=False)
    elif expr.startswith("上周"):
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        day_char = expr[-1]
        if day_char in weekday_map:
            target_weekday = weekday_map[day_char]
            days_since_monday = today.weekday()
            start_of_this_week = today - timedelta(days=days_since_monday)
            start_of_last_week = start_of_this_week - timedelta(days=7)
            result_date = start_of_last_week + timedelta(days=target_weekday)
        else:
            return json.dumps({"error": f"无法解析星期：{expression}"}, ensure_ascii=False)
    elif expr.startswith("下周"):
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        day_char = expr[-1]
        if day_char in weekday_map:
            target_weekday = weekday_map[day_char]
            days_since_monday = today.weekday()
            start_of_this_week = today - timedelta(days=days_since_monday)
            start_of_next_week = start_of_this_week + timedelta(days=7)
            result_date = start_of_next_week + timedelta(days=target_weekday)
        else:
            return json.dumps({"error": f"无法解析星期：{expression}"}, ensure_ascii=False)
    elif expr == "月初":
        result_date = today.replace(day=1)
    elif expr == "月末":
        next_month = today + relativedelta(months=1)
        first_of_next = next_month.replace(day=1)
        result_date = first_of_next - timedelta(days=1)
    elif expr == "上月初":
        first_of_this_month = today.replace(day=1)
        last_of_prev = first_of_this_month - timedelta(days=1)
        result_date = last_of_prev.replace(day=1)
    elif expr == "上月末":
        first_of_this_month = today.replace(day=1)
        result_date = first_of_this_month - timedelta(days=1)
    else:
        return json.dumps({"error": f"不支持的表达式：{expression}"}, ensure_ascii=False)

    return json.dumps({
        "date": result_date.strftime("%Y-%m-%d"),
        "expression": expression
    }, ensure_ascii=False)


@mcp.tool()
def is_workday(date: str) -> str:
    """判断指定日期是否为工作日（考虑中国法定节假日和调休）。

    参数 date 格式：YYYY-MM-DD
    返回 JSON: {"date": "...", "is_workday": true/false, "reason": "..."}
    """
    CHINESE_HOLIDAYS_2025 = {
        "2025-01-01": "元旦",
        "2025-01-28": "春节",
        "2025-01-29": "春节",
        "2025-01-30": "春节",
        "2025-01-31": "春节",
        "2025-02-01": "春节",
        "2025-02-02": "春节",
        "2025-02-03": "春节",
        "2025-02-04": "春节",
        "2025-04-04": "清明节",
        "2025-04-05": "清明节",
        "2025-04-06": "清明节",
        "2025-05-01": "劳动节",
        "2025-05-02": "劳动节",
        "2025-05-03": "劳动节",
        "2025-05-04": "劳动节",
        "2025-05-05": "劳动节",
        "2025-05-31": "端午节",
        "2025-06-01": "端午节",
        "2025-06-02": "端午节",
        "2025-10-01": "国庆节",
        "2025-10-02": "国庆节",
        "2025-10-03": "国庆节",
        "2025-10-04": "国庆节",
        "2025-10-05": "国庆节",
        "2025-10-06": "国庆节",
        "2025-10-07": "国庆节",
        "2025-10-08": "国庆节",
    }

    CHINESE_WORKDAY_WEEKEND_2025 = {
        "2025-01-26": "春节调休上班",
        "2025-02-08": "春节调休上班",
        "2025-04-27": "劳动节调休上班",
        "2025-09-28": "国庆节调休上班",
        "2025-10-11": "国庆节调休上班",
    }

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"error": f"日期格式错误，请使用 YYYY-MM-DD 格式：{date}"}, ensure_ascii=False)

    date_str = target_date.strftime("%Y-%m-%d")

    if date_str in CHINESE_HOLIDAYS_2025:
        return json.dumps({
            "date": date_str,
            "is_workday": False,
            "reason": f"法定节假日：{CHINESE_HOLIDAYS_2025[date_str]}"
        }, ensure_ascii=False)

    if date_str in CHINESE_WORKDAY_WEEKEND_2025:
        return json.dumps({
            "date": date_str,
            "is_workday": True,
            "reason": f"调休工作日：{CHINESE_WORKDAY_WEEKEND_2025[date_str]}"
        }, ensure_ascii=False)

    weekday = target_date.weekday()
    if weekday < 5:
        return json.dumps({
            "date": date_str,
            "is_workday": True,
            "reason": "正常工作日"
        }, ensure_ascii=False)
    else:
        day_name = "周六" if weekday == 5 else "周日"
        return json.dumps({
            "date": date_str,
            "is_workday": False,
            "reason": f"正常休息日（{day_name}）"
        }, ensure_ascii=False)


@mcp.tool()
def get_workdays(start_date: str, end_date: str) -> str:
    """计算两个日期之间的工作日天数（排除周末和中国法定节假日，考虑调休）。

    参数格式：YYYY-MM-DD
    返回 JSON: {"start": "...", "end": "...", "workday_count": N, "holiday_count": N, "weekend_count": N}
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"error": "日期格式错误，请使用 YYYY-MM-DD 格式"}, ensure_ascii=False)

    if start > end:
        start, end = end, start

    CHINESE_HOLIDAYS_2025 = {
        "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
        "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
        "2025-04-04", "2025-04-05", "2025-04-06",
        "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
        "2025-05-31", "2025-06-01", "2025-06-02",
        "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
        "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
    }

    CHINESE_WORKDAY_WEEKEND_2025 = {
        "2025-01-26", "2025-02-08", "2025-04-27",
        "2025-09-28", "2025-10-11",
    }

    workday_count = 0
    holiday_count = 0
    weekend_count = 0

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in CHINESE_HOLIDAYS_2025:
            holiday_count += 1
        elif date_str in CHINESE_WORKDAY_WEEKEND_2025:
            workday_count += 1
        elif current.weekday() < 5:
            workday_count += 1
        else:
            weekend_count += 1

        current += timedelta(days=1)

    return json.dumps({
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "total_days": (end - start).days + 1,
        "workday_count": workday_count,
        "holiday_count": holiday_count,
        "weekend_count": weekend_count
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
