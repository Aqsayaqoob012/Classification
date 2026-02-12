
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC




# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="AI-Based Resume Screening System",
    layout="wide"
)

# Hide Streamlit style (menu, header, footer)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}   /* top-right menu */
            footer {visibility: hidden;}     /* bottom footer */
            header {visibility: hidden;}     /* top header */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# =========================
# App Content
# =========================
st.title("Salary Data Analysis & Prediction App")
st.markdown("Exploratory Data Analysis, Feature Engineering & Regression Models Comparison")

# =========================
# Load Dataset
# =========================
st.header(" Load Dataset")

df = pd.read_csv("AI_Resume_Screening.csv")
st.success("Dataset Loaded Successfully!")

st.subheader(" Dataset Preview")
# Original head
df_head = df.head()

# Index 1 se start
df_head.index = df_head.index + 1

# Optional: Index column name
df_head.index.name = "S.No"

# Display in Streamlit
st.dataframe(df_head)


# =========================
# Dataset Information
# =========================



st.header("Dataset Information")

# -----------------------------
# Null Values & Data Types Table
# -----------------------------
info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes,
    "Non-Null Count": df.notnull().sum(),
    "Null Count": df.isnull().sum()
})

st.subheader("Data Types & Null Values")
st.dataframe(info_df)  # nice interactive table


st.subheader("Statistical Summary")
st.dataframe(df.describe())

# =========================
# Missing Values
# =========================
st.header(" Missing Value Analysis")

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success("No missing values found 🎉")
else:
    st.warning("Missing Values Found")

    # Convert to DataFrame for better formatting
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Count"]

    # Use st.dataframe with max_column_width
    st.dataframe(
        missing_df.style.set_table_styles(
            [{
                'selector': 'th',
                'props': [('max-width', '100px')]
            }, {
                'selector': 'td',
                'props': [('max-width', '100px')]
            }]
        )
    )          

    # =========================
# Handle Missing Values
# =========================

if 'Certifications' in df.columns:
    df['Certifications'] = df['Certifications'].fillna('Not Certified')



# =========================
# Assignment Answers
# =========================

st.header("📘 Data Understanding - Answers")

# Q2 Rows & Columns
rows, cols = df.shape

st.markdown("### ✅ 2. How many rows and columns are there?")
st.info(f"The dataset contains **{rows} rows** and **{cols} columns**.")

# Q4 Missing Values Handling
st.markdown("### ✅ 4. How did you handle missing values?")
st.success("""
Missing values in the **certificate** column were replaced with **'Not Certified'**
to maintain consistency and avoid data loss during model training.
""")

# Q5 Duplicate Records
st.markdown("### ✅ 5. Are there duplicate records?")

duplicates = df.duplicated().sum()

if duplicates > 0:
    df.drop_duplicates(inplace=True)

    st.warning(f"{duplicates} duplicate records were found and removed to improve data quality.")
else:
    st.success("No duplicate records were found in the dataset.")



st.markdown("### Missing Values After Cleaning")

remaining_nulls = df.isnull().sum().sum()

if remaining_nulls == 0:
    st.success("Dataset is now clean with no missing values ✅")
else:
    st.warning(f"There are still {remaining_nulls} missing values remaining.")




# =========================
# Outlier Detection & Treatment
# =========================

st.header("📊 Outlier Analysis")

num_cols = df.select_dtypes(include=['int64','float64']).columns

total_outliers = 0

for col in num_cols:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    total_outliers += outliers

    # ✅ Treat using Capping (BEST method)
    df[col] = df[col].clip(lower, upper)

st.success(f"✅ Total Outliers Detected and Treated: {total_outliers}")




# =========================
# Basic Analysis
# =========================
st.header("📊 Basic Analysis")

# 1️⃣ Overall Distribution of Shortlisted vs Non-Shortlisted
st.subheader("1. Distribution of Shortlisted vs Non-Shortlisted Candidates")

# Count plot using Plotly
import plotly.express as px

fig1 = px.histogram(df, x='Recruiter Decision', color='Recruiter Decision',
                    title="Shortlisted vs Rejected Candidates",
                    template="plotly_dark")

fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   showlegend=False)

st.plotly_chart(fig1, use_container_width=True)

# Insight
st.info("👉 Insight: Shows the balance of shortlisted vs rejected candidates.")


# 2️⃣ Key Numerical & Categorical Variables
st.subheader("2. Key Numerical & Categorical Variables")

# Numerical Columns
num_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'AI Score (0-100)']
st.markdown("**Numerical Columns:**")
st.write(num_cols)

# Categorical Columns
cat_cols = ['Education', 'Certifications', 'Job Role', 'Skills']
st.markdown("**Categorical Columns:**")
st.write(cat_cols)

# Optional: Show basic stats for numerical columns
st.markdown("**Statistical Summary of Numerical Features:**")
st.dataframe(df[num_cols].describe())




import plotly.express as px

st.subheader("1️⃣ Distribution of Numerical Features (Colorful Bars)")

num_cols = ['Experience (Years)',  'Projects Count', 'AI Score (0-100)']

# Custom vibrant colors for bars
custom_colors = [
    '#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1',
    '#33FFF0', '#F0FF33', '#FF8F33', '#8F33FF', '#33FF8F',
    '#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1'
]

for col in num_cols:
    st.markdown(f"### {col}")
    
    # Prepare color list for bars
    df_col = df[col].value_counts().sort_index().reset_index()
    df_col.columns = ['Value', 'Count']
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(df_col))]
    
    fig = px.bar(
        df_col,
        x='Value',
        y='Count',
        text='Count',
        color='Value',
        color_discrete_sequence=colors,
        template="plotly_dark"
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title=None,
        yaxis_title="Count",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)




import plotly.graph_objects as go
import numpy as np

st.subheader("Salary Expectation ($) Distribution ")

col = 'Salary Expectation ($)'
nbins = 20  # Number of bins

# Create histogram data
hist_values, bin_edges = np.histogram(df[col], bins=nbins)

# Assign a different color to each bin
custom_colors = [
    '#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1',
    '#33FFF0', '#F0FF33', '#FF8F33', '#8F33FF', '#33FF8F',
    '#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1',
    '#FFB347', '#C70039', '#900C3F', '#581845', '#1ABC9C'
]
bin_colors = [custom_colors[i % len(custom_colors)] for i in range(nbins)]

# Create figure
fig = go.Figure()
for i in range(nbins):
    fig.add_trace(go.Bar(
        x=[f"{bin_edges[i]:,.0f} - {bin_edges[i+1]:,.0f}"],
        y=[hist_values[i]],
        marker_color=bin_colors[i],
        name=f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}",
        text=[hist_values[i]],
        textposition='outside'
    ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis_title="Salary Range ($)",
    yaxis_title="Count",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)



import streamlit as st
import plotly.express as px

st.subheader("📊 Categorical Features Analysis (Bar Plots)")

# Categorical columns
cat_cols = ['Education', 'Certifications', 'Job Role', 'Skills']

# Custom colors
custom_colors = [
    '#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1',
    '#33FFF0', '#F0FF33', '#FF8F33', '#8F33FF', '#33FF8F'
]

for col in cat_cols:
    st.markdown(f"### {col}")

    # Step 1: Aggregate counts
    df_count = df[col].value_counts().reset_index()
    df_count.columns = ['Category', 'Count']

    # Step 2: Bar chart
    fig = px.bar(
        df_count,
        x='Category',
        y='Count',
        text='Count',
        color='Category',
        color_discrete_sequence=custom_colors,
        template='plotly_dark'
    )

    # Step 3: Layout customization
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )

    # Step 4: Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Optional insights
    if col == 'Education':
        st.info("👉 Most candidates have Bachelor's or Master's degrees.")
    elif col == 'Certifications':
        st.info("👉 Majority are 'Not Certified'.")
    elif col == 'Job Role':
        st.info("👉 Popular job roles: Software Engineer, Data Analyst, etc.")
    elif col == 'Skills':
        st.info("👉 Some skills are very common across candidates.")


from collections import Counter

st.header("📊 Top 10 Most Common Skills")

# Step 1: Split skills and flatten list
all_skills = df['Skills'].dropna().apply(lambda x: [skill.strip() for skill in x.split(',')])
flat_skills = [skill for sublist in all_skills for skill in sublist]

# Step 2: Count frequency
skill_counts = Counter(flat_skills)

# Step 3: Get top 10 skills
top_10_skills = skill_counts.most_common(10)

# Convert to dataframe for plotting
top_skills_df = pd.DataFrame(top_10_skills, columns=['Skill', 'Count'])

# Step 4: Bar chart
fig = px.bar(
    top_skills_df,
    x='Skill',
    y='Count',
    text='Count',
    color='Count',
    color_continuous_scale='Viridis',
    template='plotly_dark'
)

fig.update_traces(textposition='outside')
fig.update_layout(
    xaxis_title="Skill",
    yaxis_title="Number of Candidates",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Step 5: Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Optional Insight
st.info("👉 These are the 10 most common skills among all candidates. Useful for recruiter decision-making")



st.header("📊 Bivariate Analysis")

# Color map for Recruiter Decision
decision_colors = {'Shortlisted':'#33FF57', 'Rejected':'#FF5733'}

# 1️⃣ AI Score vs Recruiter Decision
st.subheader("1. AI Score by Recruiter Decision")
fig1 = px.box(df, x='Recruiter Decision', y='AI Score (0-100)',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig1, use_container_width=True, key="ai_score_box")
st.info("👉 Shortlisted candidates generally have higher AI Scores than rejected candidates.")

# 2️⃣ Experience vs Recruiter Decision
st.subheader("2. Experience vs Recruiter Decision")
fig2 = px.box(df, x='Recruiter Decision', y='Experience (Years)',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig2, use_container_width=True, key="experience_box")
st.info("👉 Shortlisted candidates tend to have slightly more experience.")

# 3️⃣ Education vs Shortlisting
st.subheader("3. Education Level vs Recruiter Decision")
edu_count = df.groupby(['Education','Recruiter Decision']).size().reset_index(name='Count')
fig3 = px.bar(edu_count, x='Education', y='Count', color='Recruiter Decision',
             barmode='group', template='plotly_dark')
st.plotly_chart(fig3, use_container_width=True, key="education_bar")
st.info("👉 Higher education levels (Bachelor, Master) have higher shortlisting rates.")

# 4️⃣ Number of Skills vs Decision
st.subheader("4. Number of Skills vs Recruiter Decision")
df['Num_Skills'] = df['Skills'].apply(lambda x: len(str(x).split(',')))
fig4 = px.box(df, x='Recruiter Decision', y='Num_Skills',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig4, use_container_width=True, key="skills_box")
st.info("👉 Candidates with more skills are more likely to be shortlisted.")

# 5️⃣ Projects Count vs Decision
st.subheader("5. Projects Count vs Recruiter Decision")
fig5 = px.box(df, x='Recruiter Decision', y='Projects Count',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig5, use_container_width=True, key="projects_box")
st.info("👉 Candidates with more projects are slightly more likely to be shortlisted.")

# 6️⃣ Salary Expectation vs Decision
st.subheader("6. Salary Expectation ($) vs Recruiter Decision")
fig6 = px.box(df, x='Recruiter Decision', y='Salary Expectation ($)',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig6, use_container_width=True, key="salary_box")
st.info("👉 Salary expectations are generally moderate; extremely high salaries reduce shortlisting probability.")

# 7️⃣ AI Score across Job Roles
st.subheader("7. AI Score by Job Role")
fig7 = px.box(df, x='Job Role', y='AI Score (0-100)',
             color='Job Role', template='plotly_dark')
st.plotly_chart(fig7, use_container_width=True, key="ai_jobrole_box")
st.info("👉 Certain job roles have higher AI Scores; e.g., Data Scientists may score higher than others.")

# 8️⃣ Correlation: Experience vs AI Score
st.subheader("8. Correlation: Experience vs AI Score")
fig8 = px.scatter(df, x='Experience (Years)', y='AI Score (0-100)',
                 color='Recruiter Decision', color_discrete_map=decision_colors,
                 template='plotly_dark')
st.plotly_chart(fig8, use_container_width=True, key="exp_vs_ai_scatter")
corr = df['Experience (Years)'].corr(df['AI Score (0-100)'])
st.info(f"👉 Correlation between Experience and AI Score: {corr:.2f}")

# 9️⃣ Certifications vs Shortlisted
st.subheader("9. Certifications vs Recruiter Decision")
cert_count = df.groupby(['Certifications','Recruiter Decision']).size().reset_index(name='Count')
fig9 = px.bar(cert_count, x='Certifications', y='Count', color='Recruiter Decision',
             barmode='group', template='plotly_dark')
st.plotly_chart(fig9, use_container_width=True, key="certifications_bar")
st.info("👉 Certain certifications are more common among shortlisted candidates.")

# 10️⃣ Education vs AI Score
st.subheader("10. Education vs AI Score")
fig10 = px.box(df, x='Education', y='AI Score (0-100)',
             color='Education', template='plotly_dark')
st.plotly_chart(fig10, use_container_width=True, key="edu_vs_ai_box")
st.info("👉 Higher education generally correlates with higher AI Scores.")

# 11️⃣ Experience across Job Roles
st.subheader("11. Experience vs Job Role")
fig11 = px.box(df, x='Job Role', y='Experience (Years)',
             color='Job Role', template='plotly_dark')
st.plotly_chart(fig11, use_container_width=True, key="exp_jobrole_box")
st.info("👉 Some job roles require more experience than others.")

# 12️⃣ Salary Outliers Impact
st.subheader("12. Salary Expectation Outliers")
fig12 = px.box(df, y='Salary Expectation ($)', template='plotly_dark')
st.plotly_chart(fig12, use_container_width=True, key="salary_outlier_box")
st.info("👉 Extreme salary expectations are capped; very high salaries reduce shortlisting probability.")

# 13️⃣ Projects Count by Recruiter Decision
st.subheader("13. Projects Count vs Recruiter Decision")
fig13 = px.box(df, x='Recruiter Decision', y='Projects Count',
             color='Recruiter Decision', color_discrete_map=decision_colors,
             template='plotly_dark')
st.plotly_chart(fig13, use_container_width=True, key="projects_decision_box")
st.info("👉 More projects slightly increase chance of shortlisting.")

# 14️⃣ Correlation Matrix (Numerical Features)
st.subheader("14. Correlation Matrix")
num_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'AI Score (0-100)', 'Num_Skills']
corr_matrix = df[num_cols].corr()
fig14 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', template='plotly_dark')
st.plotly_chart(fig14, use_container_width=True, key="corr_matrix")
st.info("👉 AI Score is most correlated with shortlisting. Projects and experience have moderate correlation.")

# 15️⃣ Overall Patterns for Shortlisting
st.subheader("15. Overall Patterns")
st.info("""
- Higher AI Scores strongly predict shortlisting.  
- More experience and higher education slightly improve chances.  
- Candidates with more skills and projects are more likely to be shortlisted.  
- Salary expectation outliers reduce probability of shortlisting.  
- Relevant certifications and popular job roles slightly favor shortlisting.
""")

st.title("🔍 Machine Learning Model Comparison App")

# Drop unnecessary columns
df = df.drop(["Resume_ID", "Name", "AI Score (0-100)"], axis=1)


target = "Recruiter Decision"

X = df.drop(target, axis=1)
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC()
}

results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Train Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, average="weighted")
    train_rec = recall_score(y_train, y_train_pred, average="weighted")
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")

    # Test Metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average="weighted")
    test_rec = recall_score(y_test, y_test_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    # Overfitting / Underfitting Logic
    gap = train_f1 - test_f1

    if train_f1 < 0.70 and test_f1 < 0.70:
        status = "Underfitting"
    elif gap > 0.10:
        status = "Overfitting"
    else:
        status = "Generalized"

    results.append([
        name,
        train_acc, train_prec, train_rec, train_f1,
        test_acc, test_prec, test_rec, test_f1,
        status
    ])

# Create DataFrame
results_df = pd.DataFrame(results, columns=[
    "Model",
    "Train Accuracy", "Train Precision", "Train Recall", "Train F1-Score",
    "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score",
    "Status"
])

# Sort by Test F1
results_df = results_df.sort_values(by="Test F1-Score", ascending=False)

st.header("📋 Model Performance Comparison")
st.dataframe(results_df)

# Highlight Best Model
best_model = results_df.iloc[0]
st.success(f"🏆 Best Model: {best_model['Model']} ({best_model['Status']})")


st.write(y.value_counts())