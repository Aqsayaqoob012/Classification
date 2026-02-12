import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Boxes */
.box {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Headings */
h1, h2, h3 {
    color: #ffffff;
}

/* Remove graph background */
.css-1kyxreq, .stPlotlyChart, .stPyplot {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)


st.title("🤖 AI-Based Resume Screening System")
st.write("End-to-end Machine Learning pipeline for predicting candidate shortlisting.")



st.markdown('<div class="box">', unsafe_allow_html=True)

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isna().sum().sum())

st.dataframe(df.head())

st.markdown('</div>', unsafe_allow_html=True)


import plotly.express as px

fig = px.histogram(
    df,
    x="AI_Score",
    color="Shortlisted",
    template="plotly_dark"
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

st.markdown('<div class="box">', unsafe_allow_html=True)

st.subheader("AI Score Distribution")
st.plotly_chart(fig, use_container_width=True)

st.write("👉 Insight: Higher AI scores strongly correlate with shortlisted candidates.")

st.markdown('</div>', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("Experience vs Shortlisting")
    fig1 = px.box(df, x="Shortlisted", y="Experience", template="plotly_dark")
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("Education Impact")
    fig2 = px.bar(df, x="Education", color="Shortlisted", template="plotly_dark")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="box">', unsafe_allow_html=True)

st.header("🏆 Model Performance")

st.dataframe(results_df)   # jisme accuracy etc ho

st.write("""
✅ Best Model: Random Forest  
👉 Reason: Highest F1-score with balanced precision & recall.
""")

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="box">', unsafe_allow_html=True)

st.header("🔮 Predict Candidate Shortlisting")

col1, col2 = st.columns(2)

experience = col1.slider("Years of Experience", 0, 15)
ai_score = col2.slider("AI Score", 0, 100)

education = st.selectbox("Education Level", df['Education'].unique())

if st.button("Predict"):
    
    input_data = [[experience, ai_score]]  # modify according to features
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Candidate Likely to be Shortlisted")
    else:
        st.error("❌ Candidate Likely to be Rejected")

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="box">', unsafe_allow_html=True)

st.header("🌍 Real World Impact")

st.write("""
This AI-powered system helps recruiters automatically filter resumes,
reduce hiring time, eliminate bias, and focus only on high-potential candidates.
It enables faster, smarter, and data-driven hiring decisions.
""")

st.markdown('</div>', unsafe_allow_html=True)
