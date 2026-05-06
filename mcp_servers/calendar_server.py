import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calendar-tools", host="0.0.0.0", port=9001)


HOLIDAYS_2025 = {
    "2025-01-01": {"name": "元旦", "type": "public_holiday"},
    "2025-01-28": {"name": "春节", "type": "public_holiday"},
    "2025-01-29": {"name": "春节", "type": "public_holiday"},
    "2025-01-30": {"name": "春节", "type": "public_holiday"},
    "2025-01-31": {"name": "春节", "type": "public_holiday"},
    "2025-02-01": {"name": "春节", "type": "public_holiday"},
    "2025-02-02": {"name": "春节", "type": "public_holiday"},
    "2025-02-03": {"name": "春节", "type": "public_holiday"},
    "2025-02-04": {"name": "春节", "type": "public_holiday"},
    "2025-04-04": {"name": "清明节", "type": "public_holiday"},
    "2025-04-05": {"name": "清明节", "type": "public_holiday"},
    "2025-04-06": {"name": "清明节", "type": "public_holiday"},
    "2025-05-01": {"name": "劳动节", "type": "public_holiday"},
    "2025-05-02": {"name": "劳动节", "type": "public_holiday"},
    "2025-05-03": {"name": "劳动节", "type": "public_holiday"},
    "2025-05-04": {"name": "劳动节", "type": "public_holiday"},
    "2025-05-05": {"name": "劳动节", "type": "public_holiday"},
    "2025-05-31": {"name": "端午节", "type": "public_holiday"},
    "2025-06-01": {"name": "端午节", "type": "public_holiday"},
    "2025-06-02": {"name": "端午节", "type": "public_holiday"},
    "2025-10-01": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-02": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-03": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-04": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-05": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-06": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-07": {"name": "国庆节", "type": "public_holiday"},
    "2025-10-08": {"name": "国庆节", "type": "public_holiday"},
}

WORKDAY_WEEKEND_2025 = {
    "2025-01-26": {"name": "春节调休上班", "type": "adjusted_workday"},
    "2025-02-08": {"name": "春节调休上班", "type": "adjusted_workday"},
    "2025-04-27": {"name": "劳动节调休上班", "type": "adjusted_workday"},
    "2025-09-28": {"name": "国庆节调休上班", "type": "adjusted_workday"},
    "2025-10-11": {"name": "国庆节调休上班", "type": "adjusted_workday"},
}

CUSTOM_EVENTS_FILE = os.path.join(os.path.dirname(__file__), "calendar_events.json")


def _load_custom_events() -> dict:
    if os.path.exists(CUSTOM_EVENTS_FILE):
        with open(CUSTOM_EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_custom_events(events: dict):
    with open(CUSTOM_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


@mcp.tool()
def get_holidays(year: int = 2025, month: int = None) -> str:
    """查询中国法定节假日列表。可按年份或月份筛选。

    参数：
    - year: 年份，默认2025
    - month: 月份（1-12），不传则返回全年

    返回节假日列表 JSON
    """
    all_holidays = {}

    for date_str, info in HOLIDAYS_2025.items():
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj.year != year:
            continue
        if month is not None and date_obj.month != month:
            continue
        all_holidays[date_str] = info

    for date_str, info in WORKDAY_WEEKEND_2025.items():
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj.year != year:
            continue
        if month is not None and date_obj.month != month:
            continue
        all_holidays[date_str] = info

    result = []
    for date_str in sorted(all_holidays.keys()):
        info = all_holidays[date_str]
        result.append({
            "date": date_str,
            "name": info["name"],
            "type": info["type"]
        })

    return json.dumps({
        "year": year,
        "month": month,
        "count": len(result),
        "holidays": result
    }, ensure_ascii=False)


@mcp.tool()
def get_holiday_detail(date: str) -> str:
    """查询指定日期的节假日详情和调休安排。

    参数 date 格式：YYYY-MM-DD
    返回该日期是否为节假日、是否调休上班、假期安排等
    """
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"error": "日期格式错误，请使用 YYYY-MM-DD"}, ensure_ascii=False)

    date_str = target.strftime("%Y-%m-%d")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    result = {
        "date": date_str,
        "weekday": weekday_names[target.weekday()],
        "is_holiday": False,
        "is_adjusted_workday": False,
        "holiday_name": None,
        "adjustment_note": None,
    }

    if date_str in HOLIDAYS_2025:
        result["is_holiday"] = True
        result["holiday_name"] = HOLIDAYS_2025[date_str]["name"]
    elif date_str in WORKDAY_WEEKEND_2025:
        result["is_adjusted_workday"] = True
        result["adjustment_note"] = WORKDAY_WEEKEND_2025[date_str]["name"]
    elif target.weekday() >= 5:
        result["is_holiday"] = True
        result["holiday_name"] = "周末"

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_next_holiday(from_date: str = None) -> str:
    """查询距离指定日期最近的下一个法定节假日。不传 from_date 则从今天算起。

    返回节假日名称、日期、还有多少天
    """
    if from_date:
        try:
            current = datetime.strptime(from_date, "%Y-%m-%d").date()
        except ValueError:
            return json.dumps({"error": "日期格式错误，请使用 YYYY-MM-DD"}, ensure_ascii=False)
    else:
        current = datetime.now().date()

    sorted_holidays = sorted(HOLIDAYS_2025.keys())
    for date_str in sorted_holidays:
        holiday_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if holiday_date > current:
            days_away = (holiday_date - current).days
            return json.dumps({
                "holiday_name": HOLIDAYS_2025[date_str]["name"],
                "date": date_str,
                "days_away": days_away,
                "from_date": current.strftime("%Y-%m-%d")
            }, ensure_ascii=False)

    return json.dumps({"message": "2025年已无更多法定节假日"}, ensure_ascii=False)


@mcp.tool()
def add_calendar_event(
    title: str,
    date: str,
    event_type: str = "reminder",
    description: str = "",
    recurrence: str = None
) -> str:
    """添加自定义日历事件。

    参数：
    - title: 事件标题
    - date: 事件日期，格式 YYYY-MM-DD
    - event_type: 事件类型（reminder/meeting/deadline/other），默认 reminder
    - description: 事件描述，可选
    - recurrence: 重复规则（daily/weekly/monthly/yearly），不传则不重复

    返回创建结果
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"error": "日期格式错误，请使用 YYYY-MM-DD"}, ensure_ascii=False)

    events = _load_custom_events()

    event_id = f"evt_{len(events) + 1}_{date.replace('-', '')}"

    event = {
        "id": event_id,
        "title": title,
        "date": date,
        "type": event_type,
        "description": description,
        "recurrence": recurrence,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    events[event_id] = event
    _save_custom_events(events)

    return json.dumps({
        "success": True,
        "event": event
    }, ensure_ascii=False)


@mcp.tool()
def list_calendar_events(
    start_date: str = None,
    end_date: str = None,
    event_type: str = None
) -> str:
    """查询自定义日历事件列表。支持按日期范围和类型筛选。

    参数：
    - start_date: 起始日期，格式 YYYY-MM-DD，可选
    - end_date: 结束日期，格式 YYYY-MM-DD，可选
    - event_type: 事件类型筛选，可选

    返回事件列表
    """
    events = _load_custom_events()

    start = None
    end = None
    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            return json.dumps({"error": "start_date 格式错误"}, ensure_ascii=False)
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return json.dumps({"error": "end_date 格式错误"}, ensure_ascii=False)

    filtered = []
    for event in events.values():
        event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()

        if start and event_date < start:
            continue
        if end and event_date > end:
            continue
        if event_type and event.get("type") != event_type:
            continue

        filtered.append(event)

    filtered.sort(key=lambda x: x["date"])

    return json.dumps({
        "count": len(filtered),
        "events": filtered
    }, ensure_ascii=False)


@mcp.tool()
def delete_calendar_event(event_id: str) -> str:
    """删除指定的自定义日历事件。

    参数 event_id: 事件ID（添加时返回的id字段）
    """
    events = _load_custom_events()

    if event_id not in events:
        return json.dumps({"success": False, "error": f"事件不存在：{event_id}"}, ensure_ascii=False)

    deleted_event = events.pop(event_id)
    _save_custom_events(events)

    return json.dumps({
        "success": True,
        "deleted_event": deleted_event
    }, ensure_ascii=False)


@mcp.tool()
def get_monthly_overview(year: int, month: int) -> str:
    """获取指定月份的日历概览，包含工作日/休息日/节假日统计。

    参数：
    - year: 年份
    - month: 月份（1-12）

    返回该月的工作日、休息日、节假日天数及详细列表
    """
    first_day = datetime(year, month, 1).date()

    if month == 12:
        last_day = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1).date() - timedelta(days=1)

    workdays = []
    weekends = []
    holidays = []
    adjusted_workdays = []

    current = first_day
    while current <= last_day:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in HOLIDAYS_2025:
            holidays.append({"date": date_str, "name": HOLIDAYS_2025[date_str]["name"]})
        elif date_str in WORKDAY_WEEKEND_2025:
            adjusted_workdays.append({"date": date_str, "name": WORKDAY_WEEKEND_2025[date_str]["name"]})
        elif current.weekday() < 5:
            workdays.append(date_str)
        else:
            weekends.append(date_str)

        current += timedelta(days=1)

    return json.dumps({
        "year": year,
        "month": month,
        "total_days": (last_day - first_day).days + 1,
        "workday_count": len(workdays) + len(adjusted_workdays),
        "weekend_count": len(weekends),
        "holiday_count": len(holidays),
        "adjusted_workday_count": len(adjusted_workdays),
        "holidays": holidays,
        "adjusted_workdays": adjusted_workdays
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="sse")
