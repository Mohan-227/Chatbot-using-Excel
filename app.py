# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import re
from transformers import pipeline
from difflib import get_close_matches

# --- Load Hugging Face model ---
st.write("🔧 Loading Hugging Face model...")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
st.write("✅ Model loaded without API key.")

# --- App Header ---
st.set_page_config(page_title="Excel Insight Chat Assistant", layout="wide")
st.title("📊 Excel Insight Chat Assistant (Offline Mode)")
st.markdown("Upload an Excel file and ask questions about your data in natural language.")
st.write("✅ App is running.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an Excel File", type=["xlsx"])
df = None

if uploaded_file:
    st.write("📥 File received. Reading Excel...")
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    # Normalize column names
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.strip().lower()) for col in df.columns]

    # Describe columns
    column_info = {col: str(df[col].dtype) for col in df.columns}

    # --- User Query ---
    user_query = st.text_input("Ask a question about your data:")

    def guess_column(query, dtype_preference=None):
        candidates = []
        for col in df.columns:
            if dtype_preference:
                if dtype_preference == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if dtype_preference == 'categorical' and not pd.api.types.is_string_dtype(df[col]):
                    continue
            score = sum(1 for word in query.lower().split() if word in col.lower())
            candidates.append((score, col))
        candidates = sorted(candidates, reverse=True)
        return candidates[0][1] if candidates and candidates[0][0] > 0 else None

    def handle_numeric_query(query):
        lowered = query.lower()
        col = guess_column(query, dtype_preference='numeric')
        if not col:
            return None

        if "average" in lowered or "mean" in lowered:
            return f"The average of **{col}** is {df[col].mean():,.2f}"
        elif "sum" in lowered:
            return f"The sum of **{col}** is {df[col].sum():,.2f}"
        elif any(word in lowered for word in ["maximum", "highest", "max"]):
            return f"The maximum value of **{col}** is {df[col].max():,.2f}"
        elif any(word in lowered for word in ["minimum", "lowest", "min"]):
            return f"The minimum value of **{col}** is {df[col].min():,.2f}"
        return None

    def handle_entity_query(query):
        lowered = query.lower()
        if any(word in lowered for word in ["highest", "lowest", "maximum", "minimum"]):
            num_col = guess_column(query, dtype_preference='numeric')
            cat_col = guess_column("name identifier label")
            if num_col and cat_col:
                if any(word in lowered for word in ["highest", "maximum"]):
                    val = df[num_col].max()
                else:
                    val = df[num_col].min()
                result = df[df[num_col] == val][[cat_col, num_col]]
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Result", data=csv, file_name="result.csv", mime="text/csv")
                return result.to_markdown(index=False)
        return None

    def handle_filter_query(query):
        lowered = query.lower()
        if "older than" in lowered or "greater than" in lowered:
            match = re.search(r"(older|greater) than (\d+)", lowered)
            if match:
                threshold = int(match.group(2))
                col = guess_column(query, dtype_preference='numeric')
                if col:
                    filtered = df[df[col] > threshold]
                    csv = filtered.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Filtered Data", data=csv, file_name="filtered.csv", mime="text/csv")
                    return filtered.head(10).to_markdown(index=False)
        return None

    def handle_count_query(query):
        lowered = query.lower()
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                unique_values = df[col].dropna().astype(str).str.lower().unique()
                for val in unique_values:
                    if val in lowered:
                        count = df[df[col].astype(str).str.lower() == val].shape[0]
                        return f"Number of rows where **{col} = {val}**: **{count}**"

        # handle cases like "any live backlogs" (i.e. any non-empty or Yes/True values)
        for col in df.columns:
            if "backlog" in col or "live" in col or "active" in col:
                non_null_count = df[df[col].astype(str).str.lower().isin(['yes', 'true', '1']) | df[col].astype(str).str.strip().ne('')].shape[0]
                if non_null_count > 0 and all(word in lowered for word in col.split("_")):
                    return f"Number of rows with **{col} marked**: **{non_null_count}**"

        match = re.search(r"(greater|above|more|>=|over|higher) than (\d+)", lowered)
        if not match:
            match = re.search(r"(less|below|under|<=|lower) than (\d+)", lowered)

        if match:
            direction = match.group(1)
            threshold = float(match.group(2))
            col = guess_column(query, dtype_preference="numeric")
            if col:
                if direction in ["greater", "above", "more", ">=", "over", "higher"]:
                    count = df[df[col] >= threshold].shape[0]
                    return f"Number of rows where **{col} ≥ {threshold}**: **{count}**"
                else:
                    count = df[df[col] <= threshold].shape[0]
                    return f"Number of rows where **{col} ≤ {threshold}**: **{count}**"
        return None

    def handle_chart_query(query):
        lowered = query.lower()
        if "bar chart" in lowered or "bar graph" in lowered:
            col = guess_column(query, dtype_preference='numeric')
            if col:
                fig = px.bar(df, x=df.index, y=col, title=f"Bar Chart of {col}")
                st.plotly_chart(fig)
                return f"✅ Bar chart for {col}"
        elif "distribution" in lowered or "histogram" in lowered:
            col = guess_column(query, dtype_preference='numeric')
            if col:
                fig = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}")
                st.plotly_chart(fig)
                return f"✅ Distribution chart for {col}"
        elif "pie chart" in lowered:
            col = guess_column(query, dtype_preference='categorical')
            if col:
                pie_data = df[col].value_counts().reset_index()
                pie_data.columns = [col, 'count']
                fig = px.pie(pie_data, names=col, values='count', title=f"Pie Chart of {col}")
                st.plotly_chart(fig)
                return None
        elif "summary" in lowered or "overview" in lowered:
            for col in df.columns:
                if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, 'count']
                    fig = px.bar(vc, x=col, y='count', title=f"{col} Summary")
                    st.plotly_chart(fig)
            return "✅ Summary visualizations generated."
        return None

    if user_query:
        with st.spinner("Thinking..."):
            for handler in [handle_entity_query, handle_numeric_query, handle_filter_query, handle_chart_query, handle_count_query]:
                result = handler(user_query)
                if result:
                    st.markdown("### Assistant Response")
                    st.markdown(result)
                    break
            else:
                prompt = f"""
You are a data assistant. Given this dataset:

Column types:
{column_info}

Here is a sample of the dataset with {len(df.columns)} columns.

Answer the question:
{user_query}
"""
                try:
                    result = qa_pipeline(prompt, max_length=256, do_sample=False)
                    reply = result[0]['generated_text']
                    st.markdown("### Assistant Response")
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"LLM Error: {e}")
else:
    st.info("📂 Please upload an Excel file to get started.")
