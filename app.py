import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from document_processor import DocumentProcessor
from profile_generator import ProfileGenerator
from vector_store import VectorStore
from fpdf import FPDF
import re
import json
from pptx.dml.color import RGBColor
import pandas as pd
from pptx import Presentation
from io import BytesIO
import base64

from pathlib import Path


# Custom CSS for branding and layout
CUSTOM_CSS = """
<style>
/* Set initial zoom level and base font size */
html {
    zoom: 1.4;
    font-size: 18px;
}

body {
    background-color: #f8f9fa;
    font-size: 1.15rem; /* Larger base font size */
    color: #0a2c4d; /* Set default text color to blue */
}

/* Import Poppins font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

/* Additional custom font style */
@font-face {
    font-family: 'Poppins';
    font-style: normal;
    font-weight: 600;
    src: url(https://fonts.gstatic.com/s/poppins/v20/pxiByp8kv8JHgFVrLEj6Z1xlFd2JQEk.woff2) format('woff2');
}

.top-banner {
    position: fixed;
    left: 0;
    top: 0;
    width: 100vw;
    background: #0a2c4d;
    text-align: center;
    padding: 0.75rem;
    z-index: 100;
}

.banner-content {
    display: flex;
    justify-content: center;
    align-items: center;
}

.banner-text {
    color: #ffc72c !important;
    font-size: 3rem;
    font-weight: 600;
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 2.5px;
    text-align: center;
}

.top-spacer {
    height: 5.5rem;
}

.header-bar {
    background-color: #0a2c4d;
    padding: 2.5rem 2rem 2rem 2rem;
    color: white;
    border-radius: 0 0 24px 24px;
    margin-bottom: 2rem;
}

.header-title {
    font-size: 3rem;  /* Increased from 2.7rem */
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    font-size: 1.6rem;  /* Increased from 1.35rem */
    font-weight: 400;
    color: #e0e6ed;
    margin-bottom: 0.5rem;
}

.progress-cue {
    font-size: 1.4rem;  /* Increased from 1.15rem */
    color: #0a2c4d;
    margin-bottom: 2.2rem;
    font-weight: 500;
}

.section-title {
    color: #0a2c4d;
    font-size: 2.3rem;  /* Increased from 2rem */
    font-weight: 700;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}

.section-desc {
    color: #0a2c4d; /* Changed from #333 to blue */
    font-size: 1.4rem;  /* Increased from 1.1rem */
    margin-bottom: 0.7rem;
    margin-top: -0.5rem;
}

.profile-section {
    background: white;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px rgba(10,44,77,0.07);
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100vw;
    background: #0a2c4d;
    color: #ffc72c;
    text-align: right;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.23rem 2.5rem;
    letter-spacing: 1.5px;
    z-index: 100;
}

.bold-action {
    font-weight: 700;
    color: #0a2c4d;
}

.key-idea {
    color: #ffc72c;
    font-weight: 700;
}

/* Increase all Streamlit element font sizes */
.stCaption, .stMarkdown, .stTextInput, .stTextArea, .stFileUploader {
    font-size: 1.3rem !important;  /* Increased from 1.05rem */
    color: #0a2c4d !important; /* Set all text to blue */
}

/* Make buttons and form elements larger */
button, input, select, textarea, .stButton>button, .stSelectbox>div>div, [data-testid="stFileUploader"] {
    font-size: 1.3rem !important;
    color: #0a2c4d !important; /* Set all text to blue */
}

/* Make sure text in the file uploader is bigger */
[data-testid="stFileUploader"] span {
    font-size: 1.3rem !important;
    color: #0a2c4d !important; /* Set all text to blue */
}

/* Increase size of question text area */
textarea {
    min-height: 120px !important;  /* Increased from default */
    font-size: 1.3rem !important;
    color: #0a2c4d !important; /* Set all text to blue */
}

/* Make all text elements larger and blue */
p, div, span, label, a {
    font-size: 1.3rem;
    color: #0a2c4d !important; /* Set all text to blue */
}

/* Override any gray text */
[style*="color: #888"], [style*="color: rgb(136, 136, 136)"], 
[style*="color: gray"], [style*="color: #333"], [style*="color: rgb(51, 51, 51)"] {
    color: #0a2c4d !important; /* Override any inline gray styles */
}

/* Specific override for Streamlit text */
.css-nahz7x, .css-ffhzg2 {
    color: #0a2c4d !important;
}

/* Increase button size */
.stButton > button {
    padding: 0.5rem 1rem !important;
    height: auto !important;
}

/* Make expanders larger */
.streamlit-expanderHeader {
    font-size: 1.3rem !important;
    color: #0a2c4d !important; /* Set to blue */
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

<script>
// Set initial zoom level using JavaScript as a backup
document.body.style.zoom = "140%";

// Additional script to ensure text colors are applied to dynamically loaded content
document.addEventListener('DOMContentLoaded', function() {
    // Find all elements with gray text and change to blue
    const grayTexts = document.querySelectorAll('[style*="color: #888"], [style*="color: rgb(136, 136, 136)"], [style*="color: gray"], [style*="color: #333"], [style*="color: rgb(51, 51, 51)"]');
    grayTexts.forEach(element => {
        element.style.color = '#0a2c4d';
    });
});
</script>
"""

# Load environment variables
load_dotenv()

# Initialize session state
for key, default in {
    'subject_docs': [],
    'context_docs': [],
    'team_docs': [],
    'profile': None,
    'user_question': '',
    'question_answer': None,
    'reference_docs': [],
    'developer_mode': False,
    'intent': "Get an overall assessment",
    'intent_other': ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Initialize components
document_processor = DocumentProcessor()
vector_store = VectorStore()
profile_generator = ProfileGenerator()

# Load and process reference PDFs from HowToInterpretHogans/
def load_reference_docs():
    reference_folder = "HowToInterpretHogans"
    reference_texts = []
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(reference_folder, filename)
            try:
                text, metadata = document_processor.process_document(file_path)
                reference_texts.append(text)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return reference_texts

# Only load reference docs once per session
if not st.session_state.reference_docs:
    st.session_state.reference_docs = load_reference_docs()

def create_pdf(profile_text, question_answer=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    # Robust font path resolution
    BASE_DIR = Path(__file__).resolve().parent
    FONT_PATH = BASE_DIR / "fonts" / "DejaVuSans.ttf"

    pdf.add_font("DejaVu", "", str(FONT_PATH), uni=True)
    pdf.set_font("DejaVu", size=12)

    lines = profile_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # Break up very long words (over 80 chars)
        if max((len(word) for word in line.split()), default=0) > 80:
            words = [word if len(word) <= 80 else ' '.join([word[i:i+80] for i in range(0, len(word), 80)]) for word in line.split()]
            line = ' '.join(words)
        try:
            if line.startswith("###"):
                heading = line.replace("###", "").strip()
                pdf.set_font("DejaVu", "B", 14)
                pdf.cell(0, 10, heading, ln=True)
                pdf.set_font("DejaVu", size=12)
            elif re.match(r"^\s*\d+\.", line) or line.startswith("-"):
                pdf.cell(10)
                pdf.multi_cell(0, 8, line)
            elif "**" in line:
                parts = re.split(r'(\*\*.*?\*\*)', line)
                for part in parts:
                    if part.startswith("**") and part.endswith("**"):
                        pdf.set_font("DejaVu", "B", 12)
                        pdf.write(8, part[2:-2])
                        pdf.set_font("DejaVu", size=12)
                    else:
                        pdf.write(8, part)
                pdf.ln(10)
            else:
                pdf.multi_cell(0, 8, line)
        except Exception as e:
            print(f"Skipped line in PDF due to error: {e}\nLine content: {repr(line)}")
            continue


    if question_answer:
        pdf.ln(10)
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 10, "Special Question Answer:", ln=True)


        pdf.set_font("DejaVu", size=12)
        try:
            pdf.multi_cell(0, 10, question_answer)
        except Exception as e:
            print(f"Skipped question_answer in PDF due to error: {e}")

    return pdf.output(dest='S')

def clean_source_text(source_text):
    """Clean up temporary filenames in sources text and replace generic file types with meaningful document descriptions."""
    if not source_text:
        return ""
    
    # First, extract source types from the text if possible
    source_types = []
    
    # Check for Hogan references
    if "Hogan" in source_text or "hogan" in source_text:
        source_types.append("Hogan Assessment")
    
    # Check for IDI references - distinguish between the two IDI types
    if "IDI" in source_text or "idi" in source_text:
        # Check for Individual Directions Inventory
        if "directions" in source_text.lower() or "individual directions" in source_text.lower():
            source_types.append("Individual Directions Inventory")
        # Check for Intercultural Development Assessment
        elif "intercultural" in source_text.lower() or "cultural" in source_text.lower():
            source_types.append("Intercultural Development Assessment")
        # Generic IDI if we can't determine which one
        else:
            source_types.append("Assessment")
    
    # Check for 360 references
    if "360" in source_text:
        source_types.append("360° Feedback")
    
    # Check for CV references
    if "CV" in source_text or "cv" in source_text or "resume" in source_text.lower():
        source_types.append("CV/Resume")
    
    # Replace specific patterns
    # Replace Hogan temp files
    source_text = re.sub(r'tmp[a-zA-Z0-9]+\.pdf\s*\(Hogan\)', 'Hogan Assessment', source_text)
    
    # Replace Individual Directions Inventory files
    source_text = re.sub(r'tmp[a-zA-Z0-9]+\.pdf\s*\(IDI\)', 'Individual Directions Inventory', source_text)
    source_text = re.sub(r'tmp[a-zA-Z0-9]+\.pdf\s*\(Individual Directions\)', 'Individual Directions Inventory', source_text)
    
    # Replace Intercultural Development references
    source_text = re.sub(r'tmp[a-zA-Z0-9]+\.pdf\s*\(Intercultural\)', 'Intercultural Development Assessment', source_text)
    
    # Replace other temp files
    source_text = re.sub(r'tmp[a-zA-Z0-9]+\.[a-z]+', '', source_text)
    source_text = re.sub(r'tmp[a-zA-Z0-9]+', '', source_text)
    
    # Replace generic file types with more meaningful descriptions
    source_text = re.sub(r'\bPDF\b', 'Document', source_text)
    source_text = re.sub(r'\bDOCX\b', 'Document', source_text)
    source_text = re.sub(r'\bDOC\b', 'Document', source_text)
    
    # Clean up formatting
    source_text = re.sub(r'\s+', ' ', source_text)  # Multiple spaces
    source_text = re.sub(r',\s*,', ',', source_text)  # Multiple commas
    source_text = re.sub(r'\(\s*\)', '', source_text)  # Empty parentheses
    source_text = re.sub(r',\s*$', '', source_text)  # Trailing commas
    source_text = re.sub(r'^\s*,\s*', '', source_text)  # Leading commas
    source_text = re.sub(r'\(\s*,', '(', source_text)  # Commas after opening parenthesis
    source_text = re.sub(r',\s*\)', ')', source_text)  # Commas before closing parenthesis
    
    # Final cleanup
    source_text = source_text.strip()
    
    # If we've removed everything but have identified source types, use them
    if (not source_text or source_text == ',' or source_text == '()') and source_types:
        return ", ".join(source_types)
    
    # If we still have nothing, return a generic source
    if not source_text or source_text == ',' or source_text == '()':
        return "Assessment Documents"
    
    return source_text

def generate_pptx_from_json(json_data, template_path=None):
    """
    Generate a PowerPoint presentation from structured JSON data.
    Uses a template if provided, otherwise creates a new presentation.
    Maps each section to the appropriate slide in the template.
    """
    # Load template if provided, otherwise use blank
    if template_path:
        try:
            prs = Presentation(template_path)
            print(f"Using template: {template_path}")
            # Debug template info
            print(f"Template has {len(prs.slides)} slides")
            for i, slide in enumerate(prs.slides):
                print(f"Slide {i+1}: {len(slide.shapes)} shapes")
                
            # Debug theme colors
            if hasattr(prs, 'theme') and hasattr(prs.theme, 'theme_color_scheme'):
                print("\nTHEME COLORS:")
                for idx, color in enumerate(prs.theme.theme_color_scheme.colors):
                    if hasattr(color, 'rgb'):
                        r, g, b = color.rgb >> 16, (color.rgb >> 8) & 0xFF, color.rgb & 0xFF
                        print(f"  Color {idx+1}: RGB({r},{g},{b})")
                    else:
                        print(f"  Color {idx+1}: [No RGB value available]")
            else:
                print("\nNo theme color information available")
        except Exception as e:
            print(f"Error loading template: {e}")
            prs = Presentation()
    else:
        prs = Presentation()
        
    # Define our brand colors
    HEADER_COLOR_WHITE = RGBColor(255, 255, 255)  # White for headers
    BODY_COLOR_BLUE = RGBColor(10, 44, 77)        # Deep blue for body text - matches template
    
    # If using a blank presentation, create slides for each section
    if template_path is None or len(prs.slides) < 2:  # If no template or not enough slides
        for section in json_data:
            slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
            slide = prs.slides.add_slide(slide_layout)
            # Set title if possible
            title_shape = None
            for shape in slide.shapes:
                if shape.name.startswith('Title'):
                    title_shape = shape
                    break
            if title_shape and hasattr(title_shape, 'text_frame'):
                title_shape.text_frame.text = section['section']
                # Set title font to white for contrast against blue background
                for paragraph in title_shape.text_frame.paragraphs:
                    if hasattr(paragraph.font, 'color') and hasattr(paragraph.font.color, 'rgb'):
                        paragraph.font.color.rgb = HEADER_COLOR_WHITE
                    if hasattr(paragraph.font, 'bold'):
                        paragraph.font.bold = True
                    if hasattr(paragraph.font, 'size'):
                        paragraph.font.size = 32 * 12700  # 32pt for headers
            
            # Add content
            content = section['content']
            sources = section.get('sources', '')
            
            # Clean up sources to remove temporary filenames
            sources = clean_source_text(sources)
            
            content_shape = None
            for shape in slide.placeholders:
                if shape.placeholder_format.type == 1:  # MSO_PLACEHOLDER.BODY
                    content_shape = shape
                    break
            if content_shape:
                content_shape.text_frame.clear()
                for line in content.split('\n'):
                    if not line.strip():
                        continue  # skip empty lines
                    p = content_shape.text_frame.add_paragraph()
                    p.text = line.lstrip('-* ').strip()
                    if line.strip().startswith(('-', '*')):
                        p.level = 0
                        p.font.bullet = True
                    if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                        try:
                            p.font.color.rgb = RGBColor(10, 44, 77)
                        except:
                            pass
                    if hasattr(p.font, 'name'):
                        try:
                            p.font.name = "Calibri"
                        except:
                            pass
                    if hasattr(p.font, 'size'):
                        try:
                            p.font.size = 18 * 12700  # 18pt (increased from 12pt)
                        except:
                            pass
                # Add sources as a separate paragraph if present
                if sources:
                    p = content_shape.text_frame.add_paragraph()
                    p.text = f"Sources: {sources}"
                    if hasattr(p.font, 'italic'):
                        p.font.italic = True
                    if hasattr(p.font, 'name'):
                        p.font.name = "Calibri"
                    if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                        p.font.color.rgb = RGBColor(10, 44, 77)
                    if hasattr(p.font, 'size'):
                        p.font.size = 18 * 12700  # 18pt (increased from 12pt)
    else:
        # Using the template - map sections to specific slides
        # Based on the template structure shown in screenshots
        
        # Keep cover slide (slide 1) and table of contents (slide 2) as is
        
        # Define a map from section titles (from the JSON) to slide numbers in the template
        section_to_slide = {
            "Presenting Concerns and Goals": 2,     # Slide 3
            "History Snapshot": 3,                  # Slide 4
            "Behavioral Observations": 4,           # Slide 5
            "Test Results by Domain": 5,            # Slide 6
            "Integrative Case Formulation": 6,      # Slide 7
            "Diagnoses": 7,                         # Slide 8
            # For legacy support and backward compatibility:
            "Profile Summary": 2,                   # Map to Presenting Concerns (slide 3)
            "Key Strengths": 4,                     # Map to Behavioral Observations (slide 5)
            "Potential Challenges": 5,              # Map to Test Results (slide 6)
            "Psychological Style": 6,               # Map to Case Formulation (slide 7)
            "Treatment Considerations": 7,          # Map to Diagnoses (slide 8)
            "Risk Factors": 7                       # Map to Diagnoses (slide 8)
        }
        
        # Process each section from the JSON data
        print(f"Processing {len(json_data)} sections")
        for section in json_data:
            section_name = section['section']
            print(f"Processing section: '{section_name}'")
                
            content = section['content']
            sources = section.get('sources', '')
            
            # Clean up sources to remove temporary filenames
            sources = clean_source_text(sources)
            
            # Find the slide index for this section
            slide_idx = section_to_slide.get(section_name)
            if slide_idx is None:
                # Try more flexible matching for alternate section names
                for map_name, idx in section_to_slide.items():
                    # Check for strict containment first
                    if map_name.lower() in section_name.lower() or section_name.lower() in map_name.lower():
                        slide_idx = idx
                        print(f"Found slide match: '{section_name}' -> '{map_name}' (slide {idx+1})")
                        break
                
                # If still no match, try more relaxed matching using section title subset
                if (section_name.lower().startswith('present') or section_name.lower().startswith('concern')) and (idx == 2):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (Presenting Concerns and Goals)")
                elif "history" in section_name.lower() and (idx == 3):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (History Snapshot)")
                elif "observation" in section_name.lower() or "mental status" in section_name.lower() and (idx == 4):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (Behavioral Observations)")
                elif "test" in section_name.lower() or "results" in section_name.lower() and (idx == 5):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (Test Results by Domain)")
                elif "formulation" in section_name.lower() or "case" in section_name.lower() and (idx == 6):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (Integrative Case Formulation)")
                elif "diagnos" in section_name.lower() or "differential" in section_name.lower() and (idx == 7):
                    slide_idx = idx
                    print(f"Mapping '{section_name}' to slide {idx+1} (Diagnoses)")
                
            # If still no match, try keyword matching for diagnosis or test sections
            if slide_idx is None:
                if "diagnos" in section_name.lower() or "dsm" in section_name.lower() or "icd" in section_name.lower():
                    slide_idx = 7  # Map to Diagnoses
                    print(f"Keyword mapping '{section_name}' to slide 8 (Diagnoses)")
                elif "test" in section_name.lower() or "assessment" in section_name.lower() or "measure" in section_name.lower():
                    slide_idx = 5  # Map to Test Results
                    print(f"Keyword mapping '{section_name}' to slide 6 (Test Results)")
                elif "history" in section_name.lower() or "background" in section_name.lower():
                    slide_idx = 3  # Map to History
                    print(f"Keyword mapping '{section_name}' to slide 4 (History Snapshot)")
                elif "observation" in section_name.lower() or "mental status" in section_name.lower() or "mse" in section_name.lower():
                    slide_idx = 4  # Map to Behavioral Observations
                    print(f"Keyword mapping '{section_name}' to slide 5 (Behavioral Observations)")
            
            if slide_idx is not None and slide_idx < len(prs.slides):
                # Use existing slide from template
                slide = prs.slides[slide_idx]
                print(f"Adding content to slide {slide_idx+1} for section '{section_name}'")
                
                # IMPORTANT CHANGE: SKIP ALL TITLE MANIPULATION
                # We will leave the template titles exactly as they are
                
                # FIND OR CREATE CONTENT SHAPE
                content_shape = None
                
                # Look for existing content shapes (not the title)
                for shape in slide.shapes:
                    if hasattr(shape, 'text_frame'):
                        # Skip any shape that looks like a title (usually at the top of slide)
                        if hasattr(shape, 'top') and shape.top < 1000000:  # ~1 inch from top
                            continue
                        # Use the first non-title text shape we find for content
                        content_shape = shape
                        print(f"Found content shape in slide {slide_idx+1}")
                        break
                
                # If no content shape found, create a new textbox for content
                if not content_shape:
                    try:
                        # Create new textbox with better positioning (below title)
                        content_left = 0.5 * 914400    # 0.5 inch from left
                        content_top = 650000     # 0.5 inches from top (below title)
                        content_width = prs.slide_width - (1 * 914400)  # Full width minus 1 inch
                        content_height = prs.slide_height - content_top - (0.5 * 914400)  # From top to bottom with margin
                        
                        content_shape = slide.shapes.add_textbox(content_left, content_top, content_width, content_height)
                        print(f"Created new content textbox on slide {slide_idx+1}")
                    except Exception as e:
                        print(f"Error creating content textbox: {e}")
                
                if content_shape:
                    try:
                        # Clear existing text
                        if hasattr(content_shape, 'text_frame'):
                            content_shape.text_frame.clear()
                        
                        # Adjust position for better spacing from header
                        if hasattr(content_shape, 'top'):
                            try:
                                if content_shape.top < 650000:
                                    content_shape.top = 650000
                            except:
                                pass
                        
                        # Enable word wrap 
                        if hasattr(content_shape, 'text_frame') and hasattr(content_shape.text_frame, 'word_wrap'):
                            content_shape.text_frame.word_wrap = True
                        
                        # Add margins to text frame if possible
                        if hasattr(content_shape, 'text_frame') and hasattr(content_shape.text_frame, 'margin_top'):
                            try:
                                # Add generous margins
                                content_shape.text_frame.margin_top = 300000     # ~8mm top margin
                                content_shape.text_frame.margin_bottom = 150000  # ~4mm bottom margin
                                content_shape.text_frame.margin_left = 150000    # ~4mm left margin
                                content_shape.text_frame.margin_right = 150000   # ~4mm right margin
                            except:
                                pass
                        
                        # Add main content as bullet-aware paragraphs
                        for line in content.split('\n'):
                            if not line.strip():
                                continue
                            p = content_shape.text_frame.add_paragraph()
                            p.text = line.lstrip('-* ').strip()
                            if line.strip().startswith(('-', '*')):
                                p.level = 0
                                p.font.bullet = True
                            if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                                try:
                                    p.font.color.rgb = RGBColor(10, 44, 77)
                                except:
                                    pass
                            if hasattr(p.font, 'name'):
                                try:
                                    p.font.name = "Calibri"
                                except:
                                    pass
                            if hasattr(p.font, 'size'):
                                try:
                                    p.font.size = 18 * 12700  # 18pt (increased from 12pt)
                                except:
                                    pass
                        
                        # Add sources as a separate paragraph if present
                        if sources:
                            p = content_shape.text_frame.add_paragraph()
                            p.text = f"Sources: {sources}"
                            if hasattr(p.font, 'italic'):
                                p.font.italic = True
                            if hasattr(p.font, 'name'):
                                p.font.name = "Calibri"
                            if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                                p.font.color.rgb = RGBColor(10, 44, 77)
                            if hasattr(p.font, 'size'):
                                p.font.size = 18 * 12700  # 18pt (increased from 12pt)
                        
                        print(f"Successfully added content to slide {slide_idx+1}")
                    except Exception as e:
                        print(f"Error setting content on slide {slide_idx+1}: {e}")
            else:
                print(f"No matching slide found for section '{section_name}', creating new slide")
                # Add a new slide with appropriate formatting
                slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
                slide = prs.slides.add_slide(slide_layout)
                
                # Create proper title with blue background
                try:
                    # Create a title box with blue background
                    title_left = 0
                    title_top = 0
                    title_width = prs.slide_width
                    title_height = 0.8 * 914400  # 0.8 inches
                    
                    # Add shape for title background
                    title_shape = slide.shapes.add_textbox(title_left, title_top, title_width, title_height)
                    
                    # Apply blue background
                    if hasattr(title_shape, 'fill'):
                        title_shape.fill.solid()
                        title_shape.fill.fore_color.rgb = RGBColor(10, 44, 77)
                    
                    # Add title text
                    title_shape.text_frame.clear()
                    p = title_shape.text_frame.add_paragraph()
                    p.text = section_name
                    p.alignment = 1  # Center
                    
                    # Format title - white text
                    if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                        p.font.color.rgb = RGBColor(255,255,255)
                    if hasattr(p.font, 'bold'):
                        p.font.bold = True
                    if hasattr(p.font, 'size'):
                        p.font.size = 36 * 12700  # 36pt (increased from 32pt)
                        
                    # Apply to all runs
                    for run in p.runs:
                        if hasattr(run.font, 'color') and hasattr(run.font.color, 'rgb'):
                            run.font.color.rgb = RGBColor(255,255,255)
                        if hasattr(run.font, 'bold'):
                            run.font.bold = True
                        if hasattr(run.font, 'size'):
                            run.font.size = 36 * 12700  # 36pt (increased from 32pt)
                except Exception as e:
                    print(f"Error creating title on new slide: {e}")
                
                # Create content box
                try:
                    # Create content box
                    content_left = 0.5 * 914400    # 0.5 inch from left
                    content_top = 650000     # 1.2 inches from top (below title)
                    content_width = prs.slide_width - (1 * 914400)  # Full width minus 1 inch margins
                    content_height = prs.slide_height - content_top - (0.5 * 914400)  # To bottom with margin
                    
                    content_shape = slide.shapes.add_textbox(content_left, content_top, content_width, content_height)
                    
                    # Add margins to text frame
                    if hasattr(content_shape, 'text_frame') and hasattr(content_shape.text_frame, 'margin_top'):
                        content_shape.text_frame.margin_top = 300000     # Top margin
                        content_shape.text_frame.margin_bottom = 150000  # Bottom margin
                        content_shape.text_frame.margin_left = 150000    # Left margin
                        content_shape.text_frame.margin_right = 150000   # Right margin
                    
                    # Enable word wrap
                    if hasattr(content_shape, 'text_frame') and hasattr(content_shape.text_frame, 'word_wrap'):
                        content_shape.text_frame.word_wrap = True
                    
                    # Add content
                    p = content_shape.text_frame.add_paragraph()
                    p.text = content
                    
                    # Format content - blue text
                    if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                        p.font.color.rgb = RGBColor(10, 44, 77)
                    if hasattr(p.font, 'name'):
                        p.font.name = "Calibri"
                    if hasattr(p.font, 'size'):
                        p.font.size = 18 * 12700  # 18pt (increased from 12pt)
                        
                    # Apply to all runs
                    for run in p.runs:
                        if hasattr(run.font, 'color') and hasattr(run.font.color, 'rgb'):
                            run.font.color.rgb = RGBColor(10, 44, 77)
                        if hasattr(run.font, 'name'):
                            run.font.name = "Calibri"
                        if hasattr(run.font, 'size'):
                            run.font.size = 18 * 12700  # 18pt (increased from 12pt)
                    
                    # Add sources if available
                    if sources:
                        p = content_shape.text_frame.add_paragraph()
                        p.text = f"Sources: {sources}"
                        
                        # Format sources - italic
                        if hasattr(p.font, 'italic'):
                            p.font.italic = True
                        if hasattr(p.font, 'name'):
                            p.font.name = "Calibri"
                        if hasattr(p.font, 'color') and hasattr(p.font.color, 'rgb'):
                            p.font.color.rgb = RGBColor(10, 44, 77)
                        if hasattr(p.font, 'size'):
                            p.font.size = 18 * 12700  # 18pt (increased from 12pt)
                except Exception as e:
                    print(f"Error creating content on new slide: {e}")
    
    # Save to BytesIO
    pptx_io = BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    return pptx_io

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Add top banner
    st.markdown(
        '<div class="top-banner">'
        '<div class="banner-content">'
        '<div class="banner-text">KNOWTHEE.AI</div>'
        '</div>'
        '</div>', 
        unsafe_allow_html=True
    )
    st.markdown('<div class="top-spacer"></div>', unsafe_allow_html=True)
    
    # Remove the header-bar div and inline the content
    st.markdown(
        '<div class="progress-cue">Based on the data you provide, we will generate a <span class="key-idea">comprehensive clinical assessment</span> and tailored guidance based on what you share. By default we will generate a PowerPoint with the following sections: <br> 1. Presenting Concerns and Goals <br> 2. History Snapshot <br> 3. Behavioral Observations <br> 4. Test Results by Domain <br> 5. Integrative Case Formulation <br> 6. Diagnoses<br><br>If you have a special query, enter it below and we will specifically address it on this page, along with the PowerPoint.</div>', 
        unsafe_allow_html=True
    )

    # Keep intent functionality in session state but hide it from UI
    if 'intent' not in st.session_state:
        st.session_state['intent'] = "Get an overall assessment"

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Documents about the Individual<br> <span style="font-size:1.2rem; font-weight:400;"></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Who are we assessing? Help us get to know this patient/client. The more you share, the more comprehensive the assessment will be. <span class="bold-action">Upload</span> psychological assessments, medical records, previous diagnoses, and other clinical documentation.</div>', unsafe_allow_html=True)
    subject_docs = st.file_uploader("Upload PDF or DOCX", type=['pdf', 'docx'], accept_multiple_files=True, key="subject")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Contextual Information <span style="font-size:1.2rem; font-weight:400;"></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">What is the clinical context? What would be helpful for us to know? <span class="bold-action">Upload</span> diagnostic criteria, treatment history, assessment measures, or relevant clinical guidelines to help us conduct a thorough assessment.</div>', unsafe_allow_html=True)
    context_docs = st.file_uploader("Upload PDF or DOCX", type=['pdf', 'docx'], accept_multiple_files=True, key="context")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Clinical Consultation Question <span style="font-size:1.2rem; font-weight:400;"></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Do you have any specific clinical questions in mind? <br>Examples: "Are there any contraindications for CBT with this patient?" • "What differential diagnoses should be considered?" • "What specific risk factors should be monitored throughout treatment?"</div>', unsafe_allow_html=True)
    user_question = st.text_area(" ", height=80, key="user_question")

    if st.button("Submit"):
        all_docs = list(st.session_state.reference_docs)

        all_metadatas = []  # NEW: to collect all metadata for the report


        with st.spinner("Processing documents...... This could take about a minute, please wait."):
            if subject_docs:
                st.session_state.subject_docs = []

                st.session_state.subject_metadatas = []  # NEW

                for file in subject_docs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getvalue())
                        text, metadata = document_processor.process_document(tmp_file.name)
                        st.session_state.subject_docs.append(text)
                        st.session_state.subject_metadatas.append(metadata)
                        os.unlink(tmp_file.name)
                all_docs.extend(st.session_state.subject_docs)


                all_metadatas.extend(st.session_state.subject_metadatas)

            if context_docs:
                st.session_state.context_docs = []
                st.session_state.context_metadatas = []  # NEW

                for file in context_docs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getvalue())
                        text, metadata = document_processor.process_document(tmp_file.name)
                        st.session_state.context_docs.append(text)
                        st.session_state.context_metadatas.append(metadata)
                        os.unlink(tmp_file.name)
                all_docs.extend(st.session_state.context_docs)


                all_metadatas.extend(st.session_state.context_metadatas)

            vector_store.store_documents(all_docs)  # all_docs is now a list of strings

            with st.spinner("Generating clinical assessment...This could take a minute. Please wait."):
                st.session_state.profile = profile_generator.generate_profile(
                    vector_store.get_relevant_chunks(),
                    all_metadatas  # Pass the metadata list for the document summary
                )

            if user_question.strip():
                st.session_state.question_answer = profile_generator.answer_question(
                    vector_store.get_relevant_chunks(), user_question
                )

    if st.session_state.profile:
        # Try to parse the profile as JSON
        try:
            profile_json = json.loads(st.session_state.profile)
            df = pd.DataFrame(profile_json)
            csv_data = df.to_csv(index=False)
            
            # Success message and dataframe display removed per user request
            
            # CSV download button removed per user request
            
            # PowerPoint generation and download
            # Use Path for robust template path handling
            template_path = Path(__file__).resolve().parent / "template.pptx"
            
            # Silently check for template without showing info messages
            if not template_path.exists():
                template_path = None
                
            with st.spinner("Generating Clinical Assessment PowerPoint... This could take about a minute, please wait."):
                try:
                    # Capture logs for debugging but don't display them by default
                    import io
                    import sys
                    
                    log_capture = io.StringIO()
                    original_stdout = sys.stdout
                    sys.stdout = log_capture
                    
                    pptx_io = generate_pptx_from_json(profile_json, template_path=template_path)
                    
                    # Restore stdout and get logs
                    sys.stdout = original_stdout
                    logs = log_capture.getvalue()
                    
                    # Only show logs if developer mode is enabled (hidden feature)
                    if st.session_state.get('developer_mode', False):
                        with st.expander("PowerPoint Generation Logs"):
                            st.code(logs)
                    
                    st.download_button(
                        label="Download Clinical Assessment",
                        data=pptx_io,
                        file_name="clinical_assessment.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key="download_pptx"
                    )
                except Exception as e:
                    # Don't show the specific error, just a generic fallback message
                    # Fallback to generating PowerPoint without template
                    try:
                        pptx_io = generate_pptx_from_json(profile_json, template_path=None)
                        st.download_button(
                            label="Download Clinical Assessment",
                            data=pptx_io,
                            file_name="clinical_assessment.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_pptx"
                        )
                    except Exception:
                        st.error("Could not generate PowerPoint. Please try again later.")
        except Exception as e:
            st.error(f"Could not parse profile as JSON: {e}")
            st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-section">', unsafe_allow_html=True)
            st.write(st.session_state.profile)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.question_answer:
            st.markdown('<div class="section-title">Clinical Consultation Response</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-section">', unsafe_allow_html=True)
            st.write(st.session_state.question_answer)
            st.markdown('</div>', unsafe_allow_html=True)


        # pdf_bytes = create_pdf(st.session_state.profile, st.session_state.question_answer)
        # st.download_button(
        #     label="Export to PDF",
        #     data=pdf_bytes,
        #     file_name="psychology_profile.pdf",
        #     mime="application/pdf"
        # )


    st.markdown('<div class="footer">KNOWTHEE.AI CLINICAL ASSESSMENT</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
