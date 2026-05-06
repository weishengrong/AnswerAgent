import json
import urllib.parse
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chart-tools")


@mcp.tool()
def generate_chart(
    chart_type: str,
    labels: list,
    datasets: list,
    title: str = "",
    width: int = 600,
    height: int = 400,
    background_color: str = "white",
) -> str:
    """生成数据可视化图表，返回图表图片URL。

    支持的图表类型：
    - bar: 柱状图，适合分类数据对比
    - line: 折线图，适合趋势变化
    - pie: 饼图，适合占比分布
    - doughnut: 环形图，类似饼图
    - radar: 雷达图，适合多维度对比
    - polarArea: 极坐标面积图
    - scatter: 散点图

    参数：
    - chart_type: 图表类型（bar/line/pie/doughnut/radar/polarArea/scatter）
    - labels: X轴标签列表，如 ["研发部", "市场部", "销售部"]
    - datasets: 数据集列表，每个数据集包含 label 和 data
      示例: [{"label": "异常人数", "data": [5, 3, 8]}, {"label": "总人数", "data": [50, 40, 60]}]
    - title: 图表标题
    - width: 图表宽度（像素）
    - height: 图表高度（像素）
    - background_color: 背景颜色

    返回：图表图片URL（QuickChart.io 生成）
    """
    default_colors = [
        "rgb(54, 162, 235)",
        "rgb(255, 99, 132)",
        "rgb(75, 192, 192)",
        "rgb(255, 205, 86)",
        "rgb(153, 102, 255)",
        "rgb(255, 159, 64)",
        "rgb(201, 203, 207)",
        "rgb(46, 139, 87)",
    ]

    chart_datasets = []
    for i, ds in enumerate(datasets):
        color = default_colors[i % len(default_colors)]
        bg_color = color.replace("rgb", "rgba").replace(")", ", 0.6)")

        chart_ds = {
            "label": ds.get("label", f"数据集{i + 1}"),
            "data": ds.get("data", []),
        }

        if chart_type in ("pie", "doughnut", "polarArea"):
            chart_ds["backgroundColor"] = [default_colors[j % len(default_colors)] for j in range(len(labels))]
        else:
            chart_ds["backgroundColor"] = bg_color
            chart_ds["borderColor"] = color
            chart_ds["borderWidth"] = 2

        chart_datasets.append(chart_ds)

    config = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": chart_datasets,
        },
        "options": {
            "plugins": {
                "title": {
                    "display": bool(title),
                    "text": title,
                    "font": {"size": 16},
                },
                "legend": {
                    "display": len(chart_datasets) > 1 or chart_type in ("pie", "doughnut"),
                },
            },
            "responsive": True,
        },
    }

    config_str = json.dumps(config, ensure_ascii=False)
    encoded = urllib.parse.quote(config_str)
    chart_url = f"https://quickchart.io/chart?c={encoded}&w={width}&h={height}&bkg={background_color}&f=png"

    return json.dumps({
        "chart_url": chart_url,
        "chart_type": chart_type,
        "title": title,
        "data_points": sum(len(ds.get("data", [])) for ds in datasets),
    }, ensure_ascii=False)


@mcp.tool()
def generate_comparison_chart(
    categories: list,
    values: list,
    title: str = "",
    chart_type: str = "bar",
) -> str:
    """生成简单的对比图表（单数据集快捷方式）。

    参数：
    - categories: 分类标签列表，如 ["研发部", "市场部", "销售部"]
    - values: 对应的数值列表，如 [5, 3, 8]
    - title: 图表标题
    - chart_type: 图表类型（bar/line/pie/doughnut），默认 bar

    返回：图表图片URL
    """
    return generate_chart(
        chart_type=chart_type,
        labels=categories,
        datasets=[{"label": title or "数据", "data": values}],
        title=title,
    )


@mcp.tool()
def generate_trend_chart(
    time_labels: list,
    values: list,
    title: str = "",
    label: str = "数值",
) -> str:
    """生成趋势折线图。

    参数：
    - time_labels: 时间标签列表，如 ["1月", "2月", "3月"]
    - values: 对应的数值列表，如 [10, 25, 18]
    - title: 图表标题
    - label: 数据集标签

    返回：图表图片URL
    """
    return generate_chart(
        chart_type="line",
        labels=time_labels,
        datasets=[{"label": label, "data": values}],
        title=title,
    )


@mcp.tool()
def generate_multi_series_chart(
    categories: list,
    series_data: list,
    title: str = "",
    chart_type: str = "bar",
) -> str:
    """生成多系列对比图表。

    参数：
    - categories: 分类标签列表
    - series_data: 多个数据系列，每个包含 name 和 values
      示例: [{"name": "异常人数", "values": [5, 3, 8]}, {"name": "正常人数", "values": [45, 37, 52]}]
    - title: 图表标题
    - chart_type: 图表类型（bar/line/radar），默认 bar

    返回：图表图片URL
    """
    datasets = [{"label": s.get("name", ""), "data": s.get("values", [])} for s in series_data]
    return generate_chart(
        chart_type=chart_type,
        labels=categories,
        datasets=datasets,
        title=title,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
