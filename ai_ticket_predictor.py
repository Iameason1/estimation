import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="AI 自动票房收入预测工具", layout="wide")

st.title("🎟️ AI 自动票房收入预测工具")
st.markdown("""
本工具适用于展览/演出等场景，通过输入票种、客流、展期及五类调节因子，快速生成票房预测，并支持三情境切换、天气API接入、图表导出、AI营销预算优化和多项目横向对比。
""")

st.sidebar.header("🌦️ 情境选择")
scenario = st.sidebar.selectbox("请选择预测情境", ["悲观", "基准", "乐观"])
scenario_params = {
    "悲观": {"W": 0.75, "M": 0.9},
    "基准": {"W": 0.85, "M": 1.0},
    "乐观": {"W": 0.95, "M": 1.1},
}

# ---------- AI 营销预算建议 ----------
st.sidebar.header("🧠 营销预算 → M 系数")
marketing_budget = st.sidebar.slider("预计营销投入（万元）", 0, 100, 20)
if marketing_budget < 10:
    M = 0.9
elif marketing_budget < 30:
    M = 1.0
else:
    M = 1.1
st.sidebar.write(f"根据投入估算营销因子 M = {M}")

# ---------- 天气 API 接入 ----------
city = st.sidebar.text_input("📍 查询天气城市（用于自动调整天气因子）", "Suzhou")
use_weather_api = st.sidebar.checkbox("使用天气API自动设定天气因子", value=False)

if use_weather_api:
    try:
        weather_url = f"https://wttr.in/{city}?format=j1"
        weather_data = requests.get(weather_url).json()
        avg_temp = float(weather_data['current_condition'][0]['temp_C'])
        if avg_temp > 35:
            W = 0.7
        elif avg_temp > 30:
            W = 0.8
        elif avg_temp > 25:
            W = 0.9
        else:
            W = 1.0
        st.sidebar.success(f"当前温度 {avg_temp}°C，设置天气因子W = {W}")
    except:
        st.sidebar.warning("⚠️ 天气API获取失败，使用默认W值")
        W = scenario_params[scenario]["W"]
else:
    W = scenario_params[scenario]["W"]

# ---------- 多项目对比输入 ----------
st.sidebar.header("🔁 多项目对比")
project_name = st.sidebar.text_input("项目名称", "项目 A")

# ---------- 基础票种设定 ----------
ticket_types = ["Z1 早鸟票", "C1 单人票", "C2 双人票", "C3 亲子票", "S1 优待票"]
ticket_prices = {
    "Z1 早鸟票": st.sidebar.number_input("Z1 早鸟票价格", 10, 200, 39),
    "C1 单人票": st.sidebar.number_input("C1 单人票价格", 10, 200, 69),
    "C2 双人票": st.sidebar.number_input("C2 双人票价格", 10, 200, 99),
    "C3 亲子票": st.sidebar.number_input("C3 亲子票价格", 10, 200, 90),
    "S1 优待票": st.sidebar.number_input("S1 优待票价格", 10, 200, 45),
}
ticket_ratios = {
    k: st.sidebar.slider(f"{k} 占比", 0.0, 1.0, 0.1 if k == "Z1 早鸟票" else 0.2)
    for k in ticket_types
}

# ---------- 时间与客流 ----------
st.sidebar.header("🕒 展期与客流")
weekday_days = st.sidebar.number_input("平日天数", 0, 200, 85)
weekend_days = st.sidebar.number_input("周末天数", 0, 100, 25)
weekday_flow = st.sidebar.number_input("平日日均客流", 0, 10000, 225)
weekend_flow = st.sidebar.number_input("周末日均客流", 0, 10000, 500)

# ---------- 固定因子 ----------
T_wd = 0.9
T_we = 1.2
L = 1.0
C = 0.95

# ---------- 早鸟票是否固定 ----------
st.sidebar.header("🎯 早鸟票策略")
fixed_earlybird = st.sidebar.checkbox("固定早鸟票张数", value=True)
fixed_earlybird_qty = st.sidebar.number_input("早鸟票销售张数", 0, 100000, 3000) if fixed_earlybird else None

# ---------- 收入计算 ----------
results = []
total_income = 0

for ticket in ticket_types:
    price = ticket_prices[ticket]
    ratio = ticket_ratios[ticket]

    if ticket == "Z1 早鸟票" and fixed_earlybird:
        income = price * fixed_earlybird_qty
        results.append((ticket, income))
        total_income += income
    else:
        wd_adjust = T_wd * W * L * C * M
        we_adjust = T_we * W * L * C * M
        wd_income = price * ratio * weekday_flow * weekday_days * wd_adjust
        we_income = price * ratio * weekend_flow * weekend_days * we_adjust
        income = wd_income + we_income
        results.append((ticket, income))
        total_income += income

# ---------- 显示结果 ----------
df = pd.DataFrame(results, columns=["票种", f"{project_name} 预测收入"])
df.loc[len(df.index)] = ["总计", total_income]
st.subheader(f"📈 {project_name} | {scenario}情境预测")
st.dataframe(df, use_container_width=True)
st.bar_chart(df.set_index("票种").iloc[:-1])

# ---------- 图表导出 ----------
st.subheader("📤 图表导出")
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(df.iloc[:-1][f"{project_name} 预测收入"], labels=df.iloc[:-1]["票种"], autopct="%1.1f%%", startangle=90)
ax.axis('equal')
st.pyplot(fig)

if st.button("📥 下载数据为 CSV"):
    st.download_button("点击下载", df.to_csv(index=False), file_name=f"{project_name}_票房预测.csv", mime="text/csv")
