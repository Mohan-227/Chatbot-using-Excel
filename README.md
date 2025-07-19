Excel Insight Chatbot (LLM-powered, Offline)
A smart assistant that lets you upload any Excel spreadsheet and ask questions about your data — using simple English.

Built with Streamlit, powered by an open-source LLM (FLAN-T5), and requires no internet connection or API keys.

 What It Can Do
 Understand natural language questions

Extract insights from any .xlsx Excel file

 Automatically detect column types (e.g., names, marks, departments)
 Answer queries with:

 Text summaries (e.g., averages, counts)

 Charts (bar, pie, histograms)

 Conditional filters (e.g., scores above 90)

 Downloadable tables

 Works on any Excel schema (schema-agnostic)

 Example Use Cases
“What is the average CGPA?”

“Show the number of students with active backlogs”

“Who has the highest salary?”

“Pie chart of gender distribution”

“Bar chart of 12th percentage”

 Setup Instructions
1. Clone this repository
bash
Copy
Edit
git clone https://github.com/yourusername/excel-insight-chatbot.git
cd excel-insight-chatbot

2.pip install -r requirements.txt
3. Run the app
bash
Copy
Edit
streamlit run app.py
 Requirements
All dependencies are listed in requirements.txt and include:

streamlit (app framework)

pandas (Excel parsing)

transformers + torch (language model)

plotly, seaborn, matplotlib (charts)

 How It Works
Upload any .xlsx file

Ask a question like “average age” or “students above 90%”

The assistant:

Cleans column names

Infers column types

Answers using code logic or LLM (as fallback)

Visualizations and CSV downloads are shown if needed

 No API Key Needed
This project uses FLAN-T5 Small, an open-source transformer model, so no key or internet is required.

 Optional Enhancements You Can Add
Query history or chat memory

CSV or PDF file support

Multiple sheet parsing

LLM fallback using OpenAI (optional)

Sidebar filters and toggles

 License
This project is licensed under the MIT License.