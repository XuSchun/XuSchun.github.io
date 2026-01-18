---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

用来完成作业的个人主页，意外感觉蛮有意思，以后会不定期更新个人信息

北大人民医院研一骨科在读，每天却和神经打交道

目前研究方向：骨关节炎（OA）和椎间盘退变（IVDD）导致背根神经节敏化所引起的疼痛（捡点师兄师姐的边角料养活自己）

来自边境十八线城市亚洲丹东，某种意义上是国际化超级大都市。喜欢哆啦A梦，喜欢PVP游戏勾心斗角，曾获得东方红小学四年一班CF杯冠军，三角洲行动千万撤离，永劫无间（被）单杀职业冠军刀一挥，CS十年牢玩家，LOL14年大乱斗选手



# 📖 教育经历
- *2025.09-至今*: &nbsp;🎓 北京大学——临床医学（外科学）
- *2020.09-2025.06*: &nbsp;🎓 中国医科大学——临床医学

# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">哆啦A梦</div><img src='images/5c67838b5c4130e5a40de7527d2032403d365a01.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Biofunctionalization of Stem Cell Scaffold for Osteogenesis and Bone Regeneration](https://www.mdpi.com/2218-273X/15/12/1700)


[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=DhtAFkwAAAAJ&citation_for_view=DhtAFkwAAAAJ:ALROH1vI_8AC) <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong>

- 抱紧师兄大腿水一水 
</div>
</div>

- 小学初中期间多次在校周报上发表评论文章，** 引用：0 **

# 🎖 个人技能
- 🔬大鼠DRG提取与培养 
- 🔬间充质干细胞（MSC）相关实验操作

# 📖 课堂作业展示: "心血管疾病预测与特征可解释性分析"
- 基于随机森林与SHAP值的心血管疾病风险预测模型，实现高精度分类与深度特征可解释性分析"

Tags:
  - 机器学习
  - 数据挖掘
  - 可解释性AI
  - 医疗预测

Tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: SHAP
  - name: Statsmodels
  - name: Pandas

## 项目背景

心血管疾病是全球范围内导致死亡的主要原因之一，早期识别高风险人群对疾病预防和干预具有重要意义。本项目基于多维度健康指标数据，构建机器学习预测模型，旨在通过分析BMI、健康状况、吸烟史、年龄、性别、运动习惯等特征，准确预测个体患心血管疾病的风险概率。

项目采用随机森林算法作为核心分类器，并通过SHAP（SHapley Additive exPlanations）值方法实现模型决策过程的深度可解释性分析，为临床决策提供科学依据。

## 数据预处理与特征工程

项目使用CVD_cleaned.csv数据集，首先对类别变量进行编码处理：将二分类变量（如心脏病患病情况、吸烟史、抑郁史、关节炎、运动习惯）映射为0/1数值，对性别变量进行Male标识编码（1为男性，0为女性）。

```python
# 二分类变量编码
binary_cols = ['Heart_Disease', 'Exercise', 'Smoking_History', 'Depression', 'Arthritis']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 性别编码
df['Sex_Male'] = df['Sex'].map({'Male': 1, 'Female': 0})

# 健康状况序数编码
health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
df['Health_Score'] = df['General_Health'].map(health_map)

# 年龄类别编码
age_map = {label: idx for idx, label in enumerate(sorted(df['Age_Category'].unique()))}
df['Age_Code'] = df['Age_Category'].map(age_map)
```

特征选择聚焦于六个核心预测变量：BMI、健康评分、年龄编码、吸烟史、性别、运动习惯。数据集经过缺失值处理后，按8:2比例划分为训练集与测试集。

<div align="center">
  <img src="/images/portfolio/cvd-prediction-model/1_correlation_heatmap.png" width="80%">
  <p>图 1：特征相关性热力图</p>
</div>
*图1：各特征间的相关性分析，健康评分与BMI呈显著正相关*

<div align="center">
  <img src="/images/portfolio/cvd-prediction-model/2_target_distribution.png" width="80%">
  <p>图 2：目标变量分布</p>
</div>
*图2：数据集中心脏病患病情况的类别分布*

## 模型训练与评估

采用随机森林分类器作为预测模型，设置100棵决策树，最大深度为8，并使用类别平衡权重（class_weight='balanced'）应对数据集不平衡问题。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
```

模型在测试集上的表现通过混淆矩阵和ROC曲线进行综合评估。混淆矩阵直观展示了模型在预测健康与患病两类样本时的准确性与误判情况，ROC曲线则反映了模型在不同阈值下的分类性能。

### 2. 模型性能评估
<table border="0">
  <tr>
    <td width="50%"><img src="/images/portfolio/cvd-prediction-model/4_confusion_matrix.png" width="100%"><br><center>图 3：混淆矩阵（平衡权重）</center></td>
    <td width="50%"><img src="/images/portfolio/cvd-prediction-model/5_roc_curve.png" width="100%"><br><center>图 4：ROC 曲线</center></td>
  </tr>

  - 图3：模型预测结果与真实标签的混淆矩阵*

  - 图4：模型ROC曲线与AUC值，展示整体分类性能*

</table>


为进一步验证特征的重要性和统计显著性，项目还构建了逻辑回归模型进行对比分析：

```python
import statsmodels.formula.api as smf

logit_res = smf.logit(
    'Heart_Disease ~ BMI + Health_Score + Smoking_History + Sex_Male',
    data=df_final
).fit()
print(logit_res.summary())
```

## 模型可解释性分析

### SHAP值分析

采用SHAP值方法对随机森林模型进行可解释性分析。SHAP值基于博弈论中的Shapley值，能够量化每个特征对模型预测结果的边际贡献。

```python
import shap

# 计算SHAP值
explainer = shap.TreeExplainer(rf)
sample_X = X_test.sample(n=500, random_state=42)
shap_values = explainer.shap_values(sample_X)

# SHAP摘要图
shap.summary_plot(shap_values[1], sample_X)

# SHAP依赖图
shap.dependence_plot(
    'BMI', shap_values[1], sample_X,
    interaction_index='Health_Score'
)
```

### 3. SHAP 模型解释
<table border="0">
  <tr>
    <td width="50%"><img src="/images/portfolio/cvd-prediction-model/6_shap_summary.png" width="100%"><br><center>图 5：SHAP 特征贡献摘要</center></td>
    <td width="50%"><img src="/images/portfolio/cvd-prediction-model/7_shap_dependence_bmi.png" width="100%"><br><center>图 6：BMI 特征依赖图</center></td>
  </tr>
  - 图5：特征重要性排序与SHAP值分布，展示各特征对预测的贡献*

  - 图6：BMI与健康评分的交互作用对预测结果的影响*
</table>



### 关键发现

1. **BMI是首要风险因子**：SHAP分析显示，BMI值越高，患心血管疾病的风险越大，这与医学常识高度一致。

2. **健康状况的显著影响**：自我评估的健康评分（Health_Score）对预测结果具有重要贡献，健康状况越差，患病风险越高。

3. **吸烟史与性别因素**：吸烟史和男性性别特征均与较高的患病风险相关，符合流行病学研究结果。

4. **交互效应**：BMI与健康评分之间存在明显的交互作用，即在BMI较高的个体中，健康评分对风险预测的影响更为显著。


<div align="center">
  <img src="/images/portfolio/cvd-prediction-model/3_violin_bmi_health.png" width="80%">
  <p>图 7：BMI分布与健康状况</p>
</div>
*图7：不同健康状况类别下BMI分布的小提琴图，展示数据的分布特征*

## 总结

本项目成功构建了心血管疾病风险预测模型，通过随机森林算法实现了较高的分类准确率，并利用SHAP值方法揭示了模型的决策逻辑。研究结果表明，BMI、健康状况、吸烟史是预测心血管疾病风险的三个关键因素，为个性化健康管理提供了科学依据。

模型的可解释性分析不仅验证了结果的可靠性，也为临床医生和公共卫生决策者提供了直观、量化的风险评估工具，具有重要的应用价值。



