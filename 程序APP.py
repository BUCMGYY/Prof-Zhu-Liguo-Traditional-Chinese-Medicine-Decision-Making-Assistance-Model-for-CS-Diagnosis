import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os

# 下载字体文件
font_url = 'https://raw.githubusercontent.com/BUCMGYY/Prof-Zhu-Liguo-Traditional-Chinese-Medicine-Decision-Making-Assistance-Model-for-CS-Diagnosis/main/SimHei.ttf'  # 确保这个链接是正确的
font_path = 'SimHei.ttf'

# 检查字体文件是否已存在，若不存在则下载
if not os.path.exists(font_path):
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)

# 加载保存的随机森林模型
model = joblib.load('MLP.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "主要症状": {"type": "categorical", "options": [0, 1, 2, 3, 4]},
    "年龄": {"type": "numerical", "min": 14.0, "max": 88.0, "default": 50.0},
    "恶风寒": {"type": "categorical", "options": [0, 1]},
    "持续时间": {"type": "categorical", "options": [0, 1, 2, 3, 4]},
    "旋颈试验": {"type": "categorical", "options": [0, 1]},
    "脉象": {"type": "categorical", "options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    "舌质": {"type": "categorical", "options": [0, 1, 2, 3]},
    "诱因": {"type": "categorical", "options": [0, 1, 2, 3]},
    "颈椎屈伸运动试验": {"type": "categorical", "options": [0, 1]},
}

# Streamlit 界面
st.title("朱立国教授名老中医颈椎病辨证决策辅助模型")

# 动态生成输入项
st.header("请输入以下特征:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 定义类别映射字典
category_mapping = {
    0: "气滞血瘀",
    1: "风寒湿痹",
    2: "肝阳上亢",
    3: "肝肾亏虚",
    4: "气虚血瘀"
}

# 预测与 SHAP 可视化
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取所有类别的概率
    probabilities = predicted_proba * 100

    # 创建预测结果文本
    predicted_class_name = category_mapping[predicted_class]  # 获取中文类别名称
    text = f"根据特征值，预测类别为: {predicted_class_name}，各类别概率：\n"
    for i, prob in enumerate(probabilities):
        category_name = category_mapping[i]  # 获取中文类别名称
        text += f"{category_name}: {prob:.2f}%\n"
    # 加载字体文件
    prop = fm.FontProperties(fname=font_path)
    # 显示预测结果，使用 Matplotlib 渲染指定字体
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontproperties=prop,  # 使用加载的字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

