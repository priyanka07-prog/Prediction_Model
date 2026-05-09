import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Custom CSS for better colors
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

#Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
df = pd.read_csv("C:\\Users\\PRIYANKA\\Downloads\\archive (2)\\CardioGoodFitness.csv")

# Sidebar menu
selected = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Analysis", "Report", "Download"]
)

st.title("Cardio Fitness Income Prediction")

if selected == "Prediction":
    st.header("Income Prediction")
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 30)
        education = st.slider("Education Level", 10, 20, 15)
    
    with col2:
        usage = st.slider("Usage", 1, 7, 3)
        fitness = st.slider("Fitness Level", 1, 5, 3)
    
    miles = st.slider("Miles Run", 0, 300, 100)
    
    # Better button with spinner
    if st.button("Predict Income", type="primary"):
        with st.spinner("Predicting..."):
            time.sleep(1)  # Simulate loading
            data = pd.DataFrame([[education, age, usage, fitness, miles]], columns=["Education", "Age", "Usage", "Fitness", "Miles"])
            prediction = model.predict(data)
            st.success(f"Predicted Income: ${prediction[0]:,.2f}")
            
            # Since it's regression, no probability, but we can show confidence interval
            # For simplicity, assume some std dev
elif selected == "Analysis":
    st.header("Data Analysis & Metrics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
    with col3:
        st.metric("Average Income", f"${df['Income'].mean():,.0f}")
    with col4:
        st.metric("Average Miles", f"{df['Miles'].mean():.1f}")
    
    # Charts
    st.subheader("Income Distribution")
    fig1 = px.histogram(df, x="Income", nbins=20, title="Income Distribution")
    st.plotly_chart(fig1)
    
    st.subheader("Age vs Income")
    fig2 = px.scatter(df, x="Age", y="Income", color="Fitness", title="Age vs Income")
    st.plotly_chart(fig2)
    
    st.subheader("Usage by Product")
    fig3 = px.bar(df.groupby("Product")["Usage"].mean().reset_index(), x="Product", y="Usage", title="Average Usage by Product")
elif selected == "Report":
    st.header("Generate Report")
    
    st.write("Generate a summary report of the dataset and model.")
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            time.sleep(2)
            report = f"""
            Cardio Fitness Dataset Report
            =============================
            
            Total Records: {len(df)}
            Average Age: {df['Age'].mean():.1f}
            Average Income: ${df['Income'].mean():,.0f}
            Average Miles: {df['Miles'].mean():.1f}
            
            Model: Linear Regression
            Features: Age, Education, Usage, Fitness, Miles
            Target: Income
            
            Correlation Matrix:
            {df[['Age', 'Education', 'Usage', 'Fitness', 'Miles', 'Income']].corr()}
            """
            st.text_area("Report", report, height=300)
            
            # Download report
            st.download_button(
                label="Download Report",
                data=report,
                file_name="cardio_report.txt",
                mime="text/plain"
            )
elif selected == "Download":
    st.header("Download Data")
    
    st.write("Download the dataset or processed data.")
    
    # Download original data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Original Dataset",
        data=csv,
        file_name="CardioGoodFitness.csv",
        mime="text/csv"
    )
    
    # Download processed data
    processed_df = pd.get_dummies(df, drop_first=True)
    csv_processed = processed_df.to_csv(index=False)
    st.download_button(
        label="Download Processed Dataset",
        data=csv_processed,
        file_name="CardioGoodFitness_processed.csv",
        mime="text/csv"
    )
    