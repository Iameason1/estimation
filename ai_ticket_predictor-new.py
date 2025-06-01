import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib
from matplotlib.font_manager import FontProperties
import requests

st.set_page_config(page_title="TICKET MIND 票知", layout="wide")

# 设置中文字体
# matplotlib.font_manager.fontManager.addfont('fronts/Simhei.ttf') #临时注册新的全局字体
#font_path = 'fronts/simhei.ttf'
#font_prop = FontProperties(fname=font_path)

#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

#plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
#matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

#font_path = 'simhei.ttf'  # 确保你的环境中有一个中文字体文件，例如 simhei.ttf
#prop = font_manager.FontProperties(fname=font_path)

# 定义页面函数
def page1():
    st.title("Page 1")
    st.write("这是第一页的内容")

    # ---------- 参数设置 ----------
    city_options = ["北京", "苏州", "成都"]
    selected_city = st.sidebar.selectbox("选择城市", city_options)

    base_price = st.sidebar.slider("基准票价设定（参考城市）", 30, 150, 69)
    city_delta = {"北京": 1.2, "苏州": 1.0, "成都": 0.85}
    adjusted_price = base_price * city_delta[selected_city]
    st.sidebar.metric("建议定价（根据城市）", f"¥{adjusted_price:.2f}")

    # ---------- 客流分析 ----------
    st.subheader("👥 客流总量分析")
    weekday_days = 85
    weekend_days = 25
    weekday_flow = 225
    weekend_flow = 500
    total_weekday_visitors = (weekday_days * weekday_flow) * city_delta[selected_city]
    total_weekend_visitors = (weekend_days * weekend_flow) * city_delta[selected_city]
    total_visitors = total_weekday_visitors + total_weekend_visitors

    df_visitors = pd.DataFrame({
        "Category": ["total_weekday_visitors", "total_weekend_visitors", "total_visitors"],
        "Visitors": [total_weekday_visitors, total_weekend_visitors, total_visitors]
    })
    st.dataframe(df_visitors, use_container_width=True)
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Visitors')
    ax1.set_xticks("weekday visitors", "weekend visitors", "total visitors")
    sns.barplot(data=df_visitors, x="Category", y="Visitors", palette="coolwarm", ax=ax1)
    
    st.pyplot(fig1)
    #plt.legend(prop=prop)

    # ---------- 销量 vs 价格 关联图 ----------
    st.subheader("📉 销量-价格回归模拟")
    prices = list(range(30, 121, 10))
    conversion_rate = [0.9 - 0.005 * (p - 30) for p in prices]
    sales = [int(1000 * r) for r in conversion_rate]

    df_conv = pd.DataFrame({"价格": prices, "预估销量": sales})
    fig2, ax2 = plt.subplots()
    sns.lineplot(x="价格", y="预估销量", data=df_conv, marker="o", ax=ax2)
    ax2.set_title("票价变动对销量的影响")
    st.pyplot(fig2)
    #plt.legend(prop=prop)    

    # ---------- 多项目对比图 ----------
    st.subheader("🔁 多项目票房对比")
    projects = ["北京展", "苏州展", "成都展"]
    incomes = [500000, 320000, 280000]
    flows = [28000, 22000, 19000]
    df_compare = pd.DataFrame({
        "项目": projects,
        "票房收入": incomes,
        "总客流": flows
    })

    col1, col2 = st.columns(2)
    with col1:
        st.metric("北京票房", "¥500,000")
        st.metric("苏州票房", "¥320,000")
        st.metric("成都票房", "¥280,000")
    with col2:
        fig3, ax3 = plt.subplots()
        sns.barplot(data=df_compare, x="项目", y="票房收入", ax=ax3)
        st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.barplot(data=df_compare, x="项目", y="总客流", palette="Greens", ax=ax4)
    st.pyplot(fig4)

    # ---------- 智能报告导出 ----------
    st.subheader("📤 导出智能报告")
    if st.button("📥 下载票价-销量数据"):
        st.download_button("点击导出 CSV", df_conv.to_csv(index=False), file_name="price_conversion_data.csv")

    if st.button("📥 下载项目对比数据"):
        st.download_button("导出多项目对比", df_compare.to_csv(index=False), file_name="multi_project_comparison.csv")


def page2():
    st.title("Page 2")
    st.write("这是第二页的内容")

    # ----------------------------
    # 用户输入
    # ----------------------------
    st.sidebar.header("参数设定")

    ticket_price = st.sidebar.slider("票价 (¥)", 30, 130, 69)
    cost_per_ticket = st.sidebar.number_input("每张票成本 (¥)", 0, 100, 20)
    marketing_spend = st.sidebar.number_input("营销预算总额 (¥)", 0, 100000, 20000)
    roi_target = st.sidebar.slider("目标 ROI", 0.0, 1.0, 0.1)

    # 模拟数据 (价格 vs 销量)
    price_range = np.linspace(30, 130, 100).reshape(-1, 1)
    true_coeff = -5  # 模拟价格弹性
    true_intercept = 800
    sales = true_coeff * price_range + true_intercept + np.random.normal(0, 10, size=price_range.shape)

    model = LinearRegression().fit(price_range, sales)
    predicted_sales = model.predict(price_range)

    revenue = predicted_sales * price_range
    total_cost = cost_per_ticket * predicted_sales + marketing_spend
    roi = (revenue - total_cost) / marketing_spend

    # ----------------------------
    # 图表展示
    # ----------------------------
    st.subheader("📊 票价 ROI 曲线与销量预测")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.plot(price_range, roi, label="ROI", color="green")
    ax1.axhline(y=roi_target, color="red", linestyle="--", label=f"目标 ROI: {roi_target:.2f}")
    ax1.set_ylabel("ROI")
    ax1.set_title("票价 vs ROI 曲线")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(price_range, predicted_sales, label="预测销量", color="blue")
    ax2.set_xlabel("票价 ")
    ax2.set_ylabel("销量")
    ax2.set_title("票价 vs 销量 曲线")
    ax2.grid(True)

    st.pyplot(fig)

    # ----------------------------
    # 最优票价推荐
    # ----------------------------
    best_roi_index = np.argmax(roi)
    best_price = float(price_range[best_roi_index][0])
    best_roi = float(roi[best_roi_index])

    st.success(f"💡 建议票价为 ¥{best_price:.2f}，可实现 ROI = {best_roi:.2f}")

    # ----------------------------
    # 平均分析
    # ----------------------------
    st.subheader("📈 平均指标")
    avg_sales = np.mean(predicted_sales)
    avg_revenue = np.mean(revenue)
    avg_price = np.mean(price_range)

    col1, col2, col3 = st.columns(3)
    col1.metric("平均票价", f"¥{avg_price:.2f}")
    col2.metric("平均销量", f"{avg_sales:.0f} 张")
    col3.metric("平均客单价", f"¥{avg_revenue / avg_sales:.2f}")

    # ----------------------------
    # 下载数据
    # ----------------------------
    df_export = pd.DataFrame({
        "票价": price_range.flatten(),
        "预测销量": predicted_sales.flatten(),
        "总收入": revenue.flatten(),
        "总成本": total_cost.flatten(),
        "ROI": roi.flatten()

    })
    st.download_button("📥 下载数据 (CSV)", data=df_export.to_csv(index=False), file_name="roi_price_sales_curve.csv")

def page3():
    st.title("Page 3")
    st.write("这是第三页的内容")


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
        st.download_button("点击下载", df.to_csv(index=False), file_name=f"{project_name}_票房预测.csv",
                           mime="text/csv")


# 创建一个侧边栏菜单
page_names_to_funcs = {
    "首页": page1,
    "页面2": page2,
    "页面3": page3,
}

selected_page = st.sidebar.selectbox("选择页面", page_names_to_funcs.keys())

# 根据选择调用相应的页面函数
page_names_to_funcs[selected_page]()

