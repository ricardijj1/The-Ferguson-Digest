# app.py
import streamlit as st
from openai import OpenAI
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from io import StringIO
import pdfplumber
from dotenv import load_dotenv

# ---- PAGE SETUP ----

# ---- PAGE SETUP ----
import streamlit as st
import os

st.set_page_config(
    page_title="Ferguson Digest", 
    page_icon="logo.png", 
    layout="wide"
)

# Display logo at the top
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Add icon next to title below the logo - aligned and close
icon_path = "Checklist--Streamline-Ultimate (1).png"
if os.path.exists(icon_path):
    import base64
    with open(icon_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{img_base64}" width="40" height="40" style="object-fit: contain;">
            <h1 style="margin: 0; line-height: 1;">Ferguson Digest</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.title("Ferguson Digest")

# ---- SIDEBAR SETUP ----
st.sidebar.title("API Setup")

# Try to get API key from environment first, then from sidebar input
load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Secret key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    st.stop()

# Department Configuration
DEPARTMENTS = [
    "Administration", "Finance", "Human Resources", "IT", "Operations", 
    "Marketing", "Sales", "Customer Service", "Legal", "Facilities",
    "Planning", "Public Works", "Parks & Recreation", "Fire Department",
    "Police Department", "Water/Sewer", "Building/Zoning"
]

# ---- HELPER FUNCTIONS ----
def extract_events_with_ai(text):
    """Extract events and dates from text using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Extract all events, meetings, deadlines, and important dates from the text. 
                Return ONLY a valid JSON array with this exact format:
                [
                  {
                    "title": "Event name",
                    "date": "YYYY-MM-DD",
                    "time": "HH:MM" or null,
                    "description": "Brief description",
                    "type": "meeting|deadline|event"
                  }
                ]
                
                If no events are found, return an empty array: []
                Do not include any other text or explanations."""},
                {"role": "user", "content": text}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        events_json = response.choices[0].message.content.strip()
        events_json = events_json.replace("```json", "").replace("```", "").strip()
        
        try:
            events = json.loads(events_json)
            return events if isinstance(events, list) else []
        except json.JSONDecodeError:
            st.warning("AI returned invalid JSON format for events")
            return []
            
    except Exception as e:
        st.error(f"Error extracting events: {str(e)}")
        return []

def analyze_departments_and_actions(text):
    """Analyze which departments are mentioned and what actions they need to take"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""Analyze this internal message and identify:
                1. Which departments are mentioned or need to be involved
                2. What specific actions each department needs to take
                
                Available departments: {', '.join(DEPARTMENTS)}
                
                Return ONLY a valid JSON object with this exact format:
                {{
                  "mentioned_departments": ["dept1", "dept2"],
                  "department_actions": {{
                    "dept1": ["action1", "action2"],
                    "dept2": ["action1"]
                  }}
                }}
                
                Use only departments from the provided list."""},
                {"role": "user", "content": text}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        analysis_json = response.choices[0].message.content.strip()
        analysis_json = analysis_json.replace("```json", "").replace("```", "").strip()
        
        try:
            analysis = json.loads(analysis_json)
            return analysis
        except json.JSONDecodeError:
            st.warning("AI returned invalid JSON format for department analysis")
            return {}
            
    except Exception as e:
        st.error(f"Error analyzing departments: {str(e)}")
        return {}

def generate_department_summary(text, department):
    """Generate a department-specific summary"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""Create a focused summary of this message specifically for the {department} department. 
                Highlight:
                - What {department} needs to know
                - Actions {department} needs to take
                - Deadlines relevant to {department}
                - How this impacts {department}'s work
                
                Keep it concise and actionable."""},
                {"role": "user", "content": text}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating {department} summary: {str(e)}")
        return f"Error generating summary for {department}"

def load_analysis_file(uploaded_file):
    """Load different file types for industry analysis"""
    try:
        if uploaded_file.type == "text/csv":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            return pd.read_csv(stringio), "csv"
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            return text, "pdf"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            return text, "txt"
        return None, "unsupported"
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, "error"

def generate_data_insights(data, file_type):
    """Generate AI insights for data analysis"""
    try:
        if file_type == "csv":
            sample_data = data.head(10).to_string(index=False)
            data_info = f"""
            Dataset Shape: {data.shape}
            Columns: {', '.join(data.columns.tolist())}
            Sample Data:
            {sample_data}
            """
            prompt = f"""Analyze this dataset and provide key insights focusing on:
            - Important trends and patterns
            - Notable data points
            - Recommendations for action
            
            Dataset Information:
            {data_info}"""
            
        else:
            text_sample = data[:3000] if len(data) > 3000 else data
            prompt = f"""Analyze this document and provide:
            - Key trends mentioned
            - Important statistics or findings
            - Main recommendations
            
            Document Content:
            {text_sample}"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Provide clear, actionable insights that help understand trends and make informed decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return "Unable to generate insights at this time."

def generate_google_calendar_url(event):
    """Generate Google Calendar add event URL"""
    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    
    title = event.get('title', 'Untitled Event')
    date_str = event.get('date', '')
    time_str = event.get('time', '')
    
    if date_str:
        try:
            event_date = datetime.strptime(date_str, '%Y-%m-%d')
            if time_str:
                event_time = datetime.strptime(time_str, '%H:%M').time()
                start_datetime = datetime.combine(event_date.date(), event_time)
                end_datetime = start_datetime + timedelta(hours=1)
                dates = f"{start_datetime.strftime('%Y%m%dT%H%M%S')}/{end_datetime.strftime('%Y%m%dT%H%M%S')}"
            else:
                dates = event_date.strftime('%Y%m%d')
        except ValueError:
            dates = ""
    else:
        dates = ""
    
    description = event.get('description', '')
    
    params = {
        'text': title,
        'dates': dates,
        'details': description
    }
    
    url_params = "&".join([f"{k}={v.replace(' ', '%20').replace('&', '%26')}" for k, v in params.items() if v])
    return f"{base_url}&{url_params}"

def generate_ics_content(events):
    """Generate ICS file content for calendar import"""
    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Ferguson Digest//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
"""
    
    for i, event in enumerate(events):
        title = event.get('title', 'Untitled Event')
        date_str = event.get('date', '')
        time_str = event.get('time', '')
        description = event.get('description', '')
        
        if date_str:
            try:
                event_date = datetime.strptime(date_str, '%Y-%m-%d')
                if time_str:
                    event_time = datetime.strptime(time_str, '%H:%M').time()
                    start_datetime = datetime.combine(event_date.date(), event_time)
                    end_datetime = start_datetime + timedelta(hours=1)
                    dtstart = start_datetime.strftime('%Y%m%dT%H%M%S')
                    dtend = end_datetime.strftime('%Y%m%dT%H%M%S')
                else:
                    dtstart = event_date.strftime('%Y%m%d')
                    dtend = dtstart
                
                ics_content += f"""BEGIN:VEVENT
UID:{i}@fergusondigest.com
DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{title}
DESCRIPTION:{description}
END:VEVENT
"""
            except ValueError:
                continue
    
    ics_content += "END:VCALENDAR"
    return ics_content

# ---- MAIN APP ----
# Create tabs for different features
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Message Analysis", "Departments", "Calendar", "Q&A", "Data Analysis"])

with tab1:
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    
    if uploaded_file:
        try:
            raw_text = uploaded_file.read().decode("utf-8")
            
            if not raw_text.strip():
                st.error("File is empty.")
                st.stop()
                
            with st.expander("View Original Text"):
                st.text_area("", raw_text, height=200, disabled=True)
            
            st.session_state.message_text = raw_text
            
        except UnicodeDecodeError:
            st.error("Could not decode file. Please ensure it's a valid text file.")
            st.stop()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    if 'message_text' in st.session_state:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Summarize internal staff messages. Highlight key points, action items, and deadlines."},
                            {"role": "user", "content": st.session_state.message_text}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content
                    st.session_state.summary = summary
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("API quota exceeded. Check your OpenAI billing.")
                    else:
                        st.error(f"Error: {error_msg}")
        
        if 'summary' in st.session_state:
            st.subheader("Summary")
            st.write(st.session_state.summary)

with tab2:
    if 'message_text' in st.session_state:
        if st.button("Analyze Departments", use_container_width=True):
            with st.spinner("Analyzing..."):
                dept_analysis = analyze_departments_and_actions(st.session_state.message_text)
                st.session_state.dept_analysis = dept_analysis
        
        if 'dept_analysis' in st.session_state and st.session_state.dept_analysis:
            analysis = st.session_state.dept_analysis
            mentioned_depts = analysis.get('mentioned_departments', [])
            
            if mentioned_depts:
                for dept in mentioned_depts:
                    with st.expander(f"{dept}"):
                        actions = analysis.get('department_actions', {}).get(dept, [])
                        if actions:
                            for action in actions:
                                st.write(f"• {action}")
                        else:
                            st.write("No specific actions identified")
                        
                        if st.button(f"Generate Summary", key=f"summary_{dept}"):
                            with st.spinner("Generating summary..."):
                                dept_summary = generate_department_summary(st.session_state.message_text, dept)
                                st.session_state[f'dept_summary_{dept}'] = dept_summary
                        
                        if f'dept_summary_{dept}' in st.session_state:
                            st.write(st.session_state[f'dept_summary_{dept}'])
    else:
        st.info("Upload a message file first.")

with tab3:
    if 'message_text' in st.session_state:
        if st.button("Extract Events", use_container_width=True):
            with st.spinner("Extracting events..."):
                events = extract_events_with_ai(st.session_state.message_text)
                st.session_state.extracted_events = events
        
        if 'extracted_events' in st.session_state and st.session_state.extracted_events:
            for i, event in enumerate(st.session_state.extracted_events):
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{event.get('title', 'Untitled')}**")
                    st.write(f"{event.get('date', 'No date')} {event.get('time', '')}")
                    st.write(f"{event.get('description', 'No description')}")
                    st.write(f"Type: {event.get('type', 'event').title()}")
                
                with col2:
                    gcal_url = generate_google_calendar_url(event)
                    st.markdown(f"[Add to Calendar]({gcal_url})")
            
            if st.session_state.extracted_events:
                ics_content = generate_ics_content(st.session_state.extracted_events)
                st.download_button(
                    label="Download Calendar File (.ics)",
                    data=ics_content,
                    file_name=f"events_{datetime.now().strftime('%Y%m%d')}.ics",
                    mime="text/calendar",
                    use_container_width=True
                )
        
        elif 'extracted_events' in st.session_state:
            st.info("No events found.")
    
    else:
        st.info("Upload a message file first.")

with tab4:
    if 'message_text' in st.session_state:
        question = st.text_input("Ask a question about this message:")
        
        if question and st.button("Get Answer", use_container_width=True):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Answer questions based on the provided message content. If information isn't available, state that clearly."},
                            {"role": "user", "content": f"Message:\n{st.session_state.message_text}\n\nQuestion: {question}"}
                        ],
                        max_tokens=300,
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content
                    st.info(answer)
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("API quota exceeded. Check your OpenAI billing.")
                    else:
                        st.error(f"Error: {error_msg}")
    
    else:
        st.info("Upload a message file first.")

with tab5:
    analysis_file = st.file_uploader("Upload file for analysis", type=["csv", "pdf", "txt"], key="analysis_upload")
    
    if analysis_file:
        with st.spinner("Loading..."):
            data, file_type = load_analysis_file(analysis_file)
        
        if file_type == "csv" and data is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                numeric_count = len(data.select_dtypes(include='number').columns)
                st.metric("Numeric Columns", numeric_count)
            
            with st.expander("View Data"):
                st.dataframe(data, use_container_width=True)
            
            if not data.select_dtypes(include='number').empty:
                st.dataframe(data.describe(), use_container_width=True)
            
            if st.button("Generate Insights", use_container_width=True):
                with st.spinner("Analyzing..."):
                    insights = generate_data_insights(data, file_type)
                    st.session_state.analysis_insights = insights
            
            if 'analysis_insights' in st.session_state:
                st.markdown(st.session_state.analysis_insights)
            
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            if numeric_cols and len(numeric_cols) >= 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    y_col = st.selectbox("Bar Chart Y-axis", numeric_cols)
                    x_col = st.selectbox("Bar Chart X-axis", data.columns.tolist())
                    
                    chart_data = data.nlargest(15, y_col) if len(data) > 15 else data
                    fig_bar = px.bar(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    fig_bar.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    if len(numeric_cols) > 1:
                        y_line = st.selectbox("Line Chart Y-axis", numeric_cols, index=1)
                    else:
                        y_line = numeric_cols[0]
                    
                    x_line = st.selectbox("Line Chart X-axis", data.columns.tolist(), key="line_x")
                    
                    try:
                        line_data = data.sort_values(by=x_line)
                        fig_line = px.line(line_data, x=x_line, y=y_line, title=f"{y_line} Trend")
                        st.plotly_chart(fig_line, use_container_width=True)
                    except:
                        st.warning("Could not create line chart")
        
        elif file_type in ["pdf", "txt"] and data is not None:
            content_preview = data[:3000] if len(data) > 3000 else data
            with st.expander("View Content"):
                st.text_area("", value=content_preview, height=300, disabled=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(data))
            with col2:
                st.metric("Words", len(data.split()))
            with col3:
                st.metric("Lines", len(data.split('\n')))
            
            if st.button("Analyze Document", use_container_width=True):
                with st.spinner("Analyzing..."):
                    insights = generate_data_insights(data, file_type)
                    st.session_state.text_insights = insights
            
            if 'text_insights' in st.session_state:
                st.markdown(st.session_state.text_insights)
        
        elif file_type == "error":
            st.error("Error loading file.")
        elif file_type == "unsupported":
            st.error("Unsupported file type.")

# ---- SIDEBAR INFO ----
st.sidebar.markdown("---")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("""
- Message summaries
- Department analysis  
- Event extraction
- Message Q&A
- Data analysis
""")

