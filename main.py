import streamlit as st
import pandas as pd
from utils.ocr import PDFTextExtractor
from utils.grading import GradingSystem
from utils.batch_processor import BatchProcessor
from utils.export import ExportManager
from utils.analytics import AnalyticsDashboard
import json
import time
from threading import Event
import io
import ollama
import requests
from utils.config import Config
import os

st.set_page_config(
    page_title="Assignment Grading System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/app_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'processed_assignments' not in st.session_state:
    st.session_state.processed_assignments = []
if 'current_llm_output' not in st.session_state:
    st.session_state.current_llm_output = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

def update_llm_output(output):
    """Update the LLM output in the session state"""
    st.session_state.current_llm_output = output
    
def update_processing_status(status, file=None):
    """Update the processing status in the session state"""
    st.session_state.processing_status = status
    st.session_state.current_file = file

def display_llm_output(llm_output):
    """Display LLM output in a structured format"""
    if not llm_output:
        return

    if not any(key in llm_output for key in ['score', 'feedback', 'improvements', 'breakdown']):
        return
        
    st.markdown("""
        <div class="llm-output-container">
            <div class="llm-output-content">
    """, unsafe_allow_html=True)
    
    # Score section
    if 'score' in llm_output:
        st.markdown("""
            <div class="score-section">
                <h4>Score</h4>
                <div class="score">{}/20</div>
            </div>
        """.format(llm_output['score']), unsafe_allow_html=True)
    
    # Feedback section
    if 'feedback' in llm_output:
        st.markdown("""
            <div class="feedback-section">
                <h4>Feedback</h4>
                <div class="feedback">{}</div>
            </div>
        """.format(llm_output['feedback']), unsafe_allow_html=True)
    
    # Improvements section
    if 'improvements' in llm_output and llm_output['improvements']:
        improvements_html = """
        <div class="improvements-section">
            <h4>Suggested Improvements</h4>
            <ul class="improvements">
        """
        for improvement in llm_output['improvements']:
            improvements_html += f"<li>{improvement}</li>"
        improvements_html += "</ul></div>"
        st.markdown(improvements_html, unsafe_allow_html=True)
    
    # Breakdown section
    if 'breakdown' in llm_output and llm_output['breakdown']:
        breakdown_html = """
        <div class="breakdown-section">
            <h4>Score Breakdown</h4>
            <ul class="breakdown">
        """
        for question, score in llm_output['breakdown'].items():
            breakdown_html += f"<li><span class='question'>{question}:</span> <span class='score'>{score}</span></li>"
        breakdown_html += "</ul></div>"
        st.markdown(breakdown_html, unsafe_allow_html=True)
    
    # Close the content and container divs
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Raw output in expander (outside main container but styled consistently)
    with st.expander("Show Raw LLM Output"):
        st.json(llm_output)

def display_upload_tab(batch_processor):
    """Display the upload tab interface with sidebar controls"""
    # Add university logo at the top of the page
    st.image('university-of-alberta-logo.png', width=150)
    
    # Add the header below the logo
    st.header("📝 Assignment Processing")
    
    with st.sidebar:        
        st.subheader("📤 Upload Controls")
        
        uploaded_files = st.file_uploader(
            "Upload Student Assignments (PDF)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple PDF files for batch processing"
        )
        
        ideal_solution_file = st.file_uploader(
            "Upload Ideal Solution (PDF)",
            type=["pdf"],
            help="Upload a single PDF file containing the ideal solution"
        )
        
        grading_instructions = st.text_area(
            "Grading Instructions",
            value='''Using the provided Ideal Solution marking scheme, carefully evaluate the submitted assignment. Your grading should be detailed and based strictly on the criteria outlined in the marking scheme.
Thorough Review: Ensure that the student has attempted every question and every part of the assignment. Double-check for completeness and accuracy in their responses before proceeding with the grade.
Detailed Grading Breakdown: Provide a clear and structured breakdown of the grade, showing how each question or section contributes to the overall score out of 20. For each part, explain how well the student met the expectations outlined in the marking scheme or where they fell short.
Constructive Feedback: Offer concise, constructive feedback to the student. Highlight any errors or areas for improvement, while also acknowledging correct approaches or strong aspects of their work. Keep the feedback brief, yet informative, ensuring the student understands their mistakes and how to improve
The feedback should be as a paragraph in three lines that highlights the mistakes refer to the solution, not the student himself.
Final Grade: Present the final score out of 20, with a clear breakdown of how each component contributed to the total grade.
Use the uploaded files as the student's work to be graded.

Provide the student marks for each question and the total marks out of 20''',
            height=300
        )
        
        if uploaded_files and ideal_solution_file:
            if st.session_state.get("can_process", True) and st.session_state.selected_model:
                if st.button("Process Assignments", type="primary"):
                    process_assignments(uploaded_files, ideal_solution_file, grading_instructions, batch_processor)
            else:
                st.error("Please select a valid model before processing")
    
    if st.session_state.processing_status:
        with st.container():
            st.markdown(f"""
            <div class="status-indicator {st.session_state.processing_status.lower()}">
                <strong>Status:</strong> {st.session_state.processing_status}
                {f'<br><strong>Current File:</strong> {st.session_state.current_file}' if st.session_state.current_file else ''}
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.current_llm_output:
        st.header("Grading Results")
        display_llm_output(st.session_state.current_llm_output)
    
    if not uploaded_files:
        st.info("Upload student assignments and an ideal solution to begin grading.")

def display_results_tab():
    """Display the results tab interface"""
    st.header("📑 Grading Results")
    
    if st.session_state.processed_assignments:
        results_df = pd.DataFrame([{
            'Assignment': a['filename'],
            'Score': a['grade']['score'],
            'Status': a['status'],
            'Feedback': a['grade']['feedback'][:100] + '...' if len(a['grade']['feedback']) > 100 else a['grade']['feedback']
        } for a in st.session_state.processed_assignments])
        
        st.dataframe(
            results_df.style.highlight_max(subset=['Score'], color='lightgreen')
                          .highlight_min(subset=['Score'], color='lightpink'),
            use_container_width=True
        )
        
        st.subheader("Export Results")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            export_format = st.selectbox(
                "Choose export format:",
                ["JSON", "CSV", "Excel", "PDF"]
            )
        
        with col2:
            if st.button("Export", type="primary"):
                try:
                    if export_format == "JSON":
                        results_json = json.dumps(st.session_state.processed_assignments, indent=2)
                        st.download_button(
                            "Download JSON",
                            results_json,
                            "grading_results.json",
                            "application/json"
                        )
                    elif export_format == "CSV":
                        csv_data = ExportManager.export_to_csv(st.session_state.processed_assignments)
                        st.download_button(
                            "Download CSV",
                            csv_data,
                            "grading_results.csv",
                            "text/csv"
                        )
                    elif export_format == "Excel":
                        excel_data = ExportManager.export_to_excel(st.session_state.processed_assignments)
                        st.download_button(
                            "Download Excel",
                            excel_data,
                            "grading_results.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    elif export_format == "PDF":
                        pdf_data = ExportManager.export_to_pdf(st.session_state.processed_assignments)
                        st.download_button(
                            "Download PDF",
                            pdf_data,
                            "grading_results.pdf",
                            "application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error exporting results: {str(e)}")
    else:
        st.info("No assignments have been processed yet.")

def process_assignments(student_files, ideal_solution_file, grading_instructions, batch_processor):
    """Process uploaded assignments with real-time feedback"""
    if not ideal_solution_file:
        st.warning("Please upload an ideal solution first.")
        return False
    
    try:
        update_processing_status("Processing Ideal Solution")
        ideal_solution_bytes = ideal_solution_file.read()
        ideal_solution_result = batch_processor.text_extractor.process_pdf(ideal_solution_bytes)
        ideal_solution_text = ideal_solution_result["text"]
        
        results = []
        total_files = len(student_files)
        
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(student_files, 1):
            try:
                update_processing_status("Processing", file.name)
                progress = idx / total_files
                progress_bar.progress(progress)
                
                pdf_bytes = file.read()
                extraction_result = batch_processor.text_extractor.process_pdf(pdf_bytes)
                
                grade_result = batch_processor.grading_system.evaluate_submission(
                    extraction_result["text"],
                    ideal_solution_text,
                    grading_instructions,
                    model=st.session_state.selected_model
                )
                
                update_llm_output(grade_result)
                
                results.append({
                    "filename": file.name,
                    "text": extraction_result["text"],
                    "grade": grade_result,
                    "status": "success"
                })
                
            except Exception as e:
                update_processing_status("Error", file.name)
                results.append({
                    "filename": file.name,
                    "text": "",
                    "grade": {"score": 0, "feedback": str(e), "improvements": []},
                    "status": "error"
                })
        
        st.session_state.processed_assignments = results
        update_processing_status("Completed")
        return True
        
    except Exception as e:
        update_processing_status("Error")
        st.error(f"Error processing assignments: {str(e)}")
        return False

def main():
    """Main Streamlit application"""
    with st.sidebar:
        st.title("📝 Assignment Grading")
        
        text_extractor = PDFTextExtractor()
        grading_system = GradingSystem()
        batch_processor = BatchProcessor(text_extractor, grading_system)
        
        if 'grading_system' not in st.session_state:
            st.session_state.grading_system = grading_system
        
        st.markdown("### 🤖 Model Selection")
        
        provider = st.selectbox(
            "Choose Provider",
            ["Groq", "OpenAI", "Claude", "Local Only"],
            help="Select an AI provider"
        )

        if provider != "Local Only":
            provider_key = provider.lower()
            api_key = st.text_input(
                f"{provider} API Key",
                type="password",
                help=f"Enter your {provider} API key"
            )
            
            if st.button("🔄 Refresh Available Models"):
                if api_key:
                    st.session_state.api_keys[provider_key] = api_key
                    os.environ[f"{provider.upper()}_API_KEY"] = api_key
                    st.session_state.grading_system.refresh_available_models()
                    st.success(f"Refreshed available {provider} models")
                else:
                    st.warning(f"Please enter {provider} API key first")

            if provider_key in st.session_state.get("api_keys", {}):
                models = st.session_state.grading_system.get_available_models(provider)
                if models:
                    st.session_state.selected_model = st.selectbox(
                        f"Choose {provider} Model",
                        models,
                        help=f"Select {provider} model to use"
                    )
                    st.session_state["can_process"] = True
                else:
                    st.warning(f"No models available for {provider}")
                    st.session_state["can_process"] = False
                    st.session_state.selected_model = None
        else:
            try:
                models = grading_system.get_available_models("Local Only")
                if models:
                    st.session_state.selected_model = st.selectbox(
                        "Choose Local Model",
                        models,
                        help="Select a local model to use"
                    )
                    st.session_state["can_process"] = True
                else:
                    st.warning("No local models available. Please install models using 'ollama pull <model>'")
                    st.session_state["can_process"] = False
                    st.session_state.selected_model = None
            except Exception as e:
                st.error(f"Error fetching local models: {str(e)}")
                st.session_state["can_process"] = False
                st.session_state.selected_model = None
        
        st.markdown("---")
        current_tab = st.radio("Navigation", ["Upload", "Results", "Analytics"])
    
    if current_tab == "Upload":
        display_upload_tab(batch_processor)
    elif current_tab == "Results":
        display_results_tab()
    else:
        AnalyticsDashboard.display_dashboard(st.session_state.processed_assignments)

if __name__ == "__main__":
    main()