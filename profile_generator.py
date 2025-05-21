import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

class ProfileGenerator:
    def __init__(self):
        self.system_prompt = """You are a world-class expert in psychology, psychological assessment, and mental health. You specialize in synthesizing diverse data sources—such as psychological assessments, medical history, therapy notes, and diagnostic evaluations—into insightful, psychologically sophisticated profiles. Your goal is to produce actionable insights, grounded in evidence, that support treatment planning and patient care. Always cite the data source behind your claims and remain both rigorous and humanistic in tone."""

    def generate_profile(self, document_chunks: List[str], metadata: List[dict] = None) -> str:
        """Generate a psychology profile from document chunks and optional metadata, returning structured JSON output."""
        # Build the document type list for the LLM prompt and for the report
        doc_types = list(dict.fromkeys(meta['file_type'] for meta in metadata)) if metadata else []
        doc_type_list = "\n".join(f"- {doc_type}" for doc_type in doc_types)

        # Create a mapping of document types for cleaning up sources later
        doc_type_map = {}
        
        # Identify the types of documents based on content
        assessment_types = []
        
        # Check for Hogan assessment content
        hogan_terms = ["hogan", "hpi", "hds", "mvpi", "motives values preferences", "personality inventory", "development survey"]
        has_hogan = any(term in " ".join(document_chunks).lower() for term in hogan_terms)
        if has_hogan:
            assessment_types.append("Hogan Assessment")
            
        # Check for 360 content
        has_360 = "360" in " ".join(document_chunks) or "360-degree" in " ".join(document_chunks).lower()
        if has_360:
            assessment_types.append("360° Feedback")
            
        # Check for CV/Resume content
        cv_terms = ["cv", "resume", "résumé", "curriculum vitae", "work history", "professional experience", "education:"]
        has_cv = any(term in " ".join(document_chunks).lower() for term in cv_terms)
        if has_cv:
            assessment_types.append("CV/Resume")
            
        # Check for Intercultural Development Inventory
        intercultural_terms = ["intercultural development inventory", "intercultural sensitivity", "cultural competence"]
        has_intercultural = any(term in " ".join(document_chunks).lower() for term in intercultural_terms)
        if has_intercultural:
            assessment_types.append("Intercultural Development Assessment")
        
        # Check for Individual Directions Inventory
        individual_directions_terms = ["individual directions inventory", "idi report", "directions inventory"]
        has_directions = any(term in " ".join(document_chunks).lower() for term in individual_directions_terms)
        if has_directions:
            assessment_types.append("Individual Directions Inventory")
        
        # Check for performance reviews
        perf_terms = ["performance review", "annual review", "performance assessment", "performance rating"]
        has_perf = any(term in " ".join(document_chunks).lower() for term in perf_terms)
        if has_perf:
            assessment_types.append("Performance Review")
        
        # Check for interview notes
        interview_terms = ["interview notes", "interview summary", "candidate interview"]
        has_interview = any(term in " ".join(document_chunks).lower() for term in interview_terms)
        if has_interview:
            assessment_types.append("Interview Notes")

        if metadata:
            for meta in metadata:
                if 'file_name' in meta and 'file_type' in meta:
                    # Map temporary filenames to their document types
                    file_name = meta['file_name']
                    file_type = meta['file_type'].upper()
                    
                    # Parse meaningful document type from filename if possible
                    doc_type = None
                    
                    # Check if it's a Hogan assessment
                    if 'hogan' in file_name.lower():
                        doc_type = "Hogan Assessment"
                    # Check if it's a 360 assessment
                    elif '360' in file_name:
                        doc_type = "360° Feedback"
                    # Check if it's a CV or resume
                    elif any(term in file_name.lower() for term in ['cv', 'resume', 'résumé']):
                        doc_type = "CV/Resume"
                    # Check if it's an IDI assessment
                    elif 'idi' in file_name.lower():
                        doc_type = "IDI Assessment"
                    # Otherwise use a more generic but still descriptive term
                    else:
                        doc_type = f"{file_type} Document"
                    
                    doc_type_map[file_name] = doc_type

        # Combine detected document types with the filenames
        detected_doc_types = ", ".join(assessment_types) if assessment_types else "Submitted Documents"

        doc_summary_prompt = (
            "You have been provided with the following types of documents for your analysis:\n"
            f"{doc_type_list}\n\n"
            f"Based on content analysis, these appear to include: {detected_doc_types}\n\n"
            "EXTREMELY IMPORTANT GUIDANCE ON SOURCES:\n"
            "1. When citing sources, DO NOT refer to them by their file type (e.g., 'PDF', 'DOCX'). Instead, identify them by their content type:\n"
            "   - Refer to personality assessments as 'Hogan Assessment' or similar specific assessment name\n"
            "   - Refer to 360-degree feedback as '360° Feedback'\n"
            "   - Refer to resumes as 'CV/Resume'\n"
            "   - Refer to intercultural assessments as 'IDI Assessment'\n"
            "   - For other documents, identify them by their purpose (e.g., 'Performance Review', 'Interview Notes')\n\n"
            "2. For each major claim or insight in your analysis, include a brief in-text citation showing the source, like this: '... demonstrates strong analytical abilities (Hogan Assessment).' or '... has experience managing global teams (CV/Resume).'\n\n"
            "Use all and only the documents and data provided by the user. "
            "You must only reference the document types listed above. Do not invent or assume the existence of other data sources. "
            "If a type of data (e.g., 'Coaching Notes') is not present in the provided documents, do not reference it.\n\n"
            "For each section of your analysis, make a good faith effort to use and reference insights from all of the provided documents. \n\n"
        )
        
        # Join document chunks for context
        context = "\n\n".join(document_chunks)
        
        # Format metadata for the prompt
        metadata_text = ""
        if metadata and len(metadata) > 0:
            metadata_items = []
            for meta in metadata:
                for key, value in meta.items():
                    if key != 'file_type' and key != 'filename':
                        metadata_items.append(f"{key}: {value}")
            metadata_text = "\n".join(metadata_items)

        prompt = (
            doc_summary_prompt +
            f"Based on the following psychology documents, generate a comprehensive psychology profile:\n\n"
            f"Person Information:\n{metadata_text}\n\n"
            "IMPORTANT FORMATTING INSTRUCTIONS:\n"
            "- For 'Key Strengths', 'Potential Challenges', 'Treatment Considerations', and 'Risk Factors' sections, ALWAYS format the content as a numbered list (1., 2., 3., etc.)\n"
            "- Insert a blank line between each numbered item (double line break)\n"
            "- Each point should be focused on a single strength, challenge, or consideration\n"
            "- Limit each enumerated list to a maximum of 5 items\n"
            "- For 'Profile Summary' and 'Psychological Style' sections, use paragraph format\n"
            "- Each significant claim should include a parenthetical reference to the source (e.g., 'exhibits anxious tendencies (Psychological Assessment)')\n"
            "- Do not use markdown formatting or special characters that might interfere with JSON\n\n"
            "Sections:\n"
            "1. Profile Summary\n"
            "2. Key Strengths\n"
            "3. Potential Challenges\n"
            "4. Psychological Style\n"
            "5. Treatment Considerations\n"
            "6. Risk Factors\n\n"
            "Example output:\n"
            "[\n"
            "  {\"section\": \"Profile Summary\", \"content\": \"The patient exhibits signs of moderate anxiety with comorbid depressive features (Psychological Assessment) and has shown partial response to previous cognitive-behavioral interventions (Treatment History)...\", \"sources\": \"Psychological Assessment, Treatment History\"},\n"
            "  {\"section\": \"Key Strengths\", \"content\": \"1. Strong introspective abilities and psychological mindedness (Psychological Assessment)\\n\\n2. Consistent engagement in therapeutic process (Treatment Notes)\\n\\n3. Supportive family environment (Clinical Interview)\", \"sources\": \"Psychological Assessment, Treatment Notes, Clinical Interview\"},\n"
            "  {\"section\": \"Potential Challenges\", \"content\": \"1. Tendency toward rumination and catastrophic thinking (Psychological Assessment)\\n\\n2. Difficulty with emotional regulation during acute stress (Treatment Notes)\\n\\n3. Inconsistent application of coping strategies (Psychological Assessment)\", \"sources\": \"Psychological Assessment, Treatment Notes\"},\n"
            "  {\"section\": \"Psychological Style\", \"content\": \"The patient demonstrates an anxious-avoidant attachment style (Psychological Assessment) with a tendency to withdraw during interpersonal conflicts (Clinical Interview)...\", \"sources\": \"Psychological Assessment, Clinical Interview\"},\n"
            "  {\"section\": \"Treatment Considerations\", \"content\": \"1. Structured cognitive-behavioral approaches with emphasis on thought records (Psychological Assessment)\\n\\n2. Gradual exposure to anxiety-provoking situations (Treatment History)\\n\\n3. Mindfulness training to reduce rumination (Clinical Interview)\", \"sources\": \"Psychological Assessment, Treatment History, Clinical Interview\"},\n"
            "  {\"section\": \"Risk Factors\", \"content\": \"1. History of passive suicidal ideation during major depressive episodes (Treatment History)\\n\\n2. Social isolation during periods of heightened anxiety (Psychological Assessment)\\n\\n3. Tendency to discontinue medication without consultation (Treatment Notes)\", \"sources\": \"Treatment History, Psychological Assessment, Treatment Notes\"}\n"
            "]\n\n"
            f"{context}\n\n"
            "Return only the JSON array, with no extra commentary or explanation.\n"
            "Remember to format 'Key Strengths', 'Potential Challenges', 'Treatment Considerations', and 'Risk Factors' as numbered lists with proper line breaks between items."
        )

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )

        profile_content = response.choices[0].message.content
        
        # Clean up sources in the profile content
        try:
            import json
            profile_json = json.loads(profile_content)
            
            for section in profile_json:
                if "sources" in section:
                    sources = section["sources"]
                    
                    # Clean up temporary filenames in sources
                    # Pattern to match temporary filenames like tmp123abc.pdf
                    temp_file_pattern = re.compile(r'tmp[a-zA-Z0-9]+\.[a-z]+')
                    # Also match other temporary-looking names like tmplwgjkk8x.pdf
                    generic_temp_pattern = re.compile(r'tmp[a-zA-Z0-9]+\.pdf')
                    
                    # Replace temp filenames with their document types
                    for filename, doc_type in doc_type_map.items():
                        if filename in sources:
                            sources = sources.replace(filename, doc_type)
                    
                    # Replace any remaining temporary filenames with their file types
                    sources = temp_file_pattern.sub('Document', sources)
                    sources = generic_temp_pattern.sub('Document', sources)
                    
                    # Clean up any remaining temp files in parentheses
                    sources = re.sub(r'\(tmp[^)]*\)', '', sources)
                    
                    # Replace multiple commas with a single comma
                    sources = re.sub(r',\s*,', ',', sources)
                    # Remove trailing commas
                    sources = re.sub(r',\s*$', '', sources)
                    # Clean up whitespace
                    sources = re.sub(r'\s+', ' ', sources).strip()
                    
                    section["sources"] = sources
            
            # Convert back to JSON string
            profile_content = json.dumps(profile_json, ensure_ascii=False)
        except Exception as e:
            # If any error occurs during cleaning, return the original content
            print(f"Error cleaning up sources: {e}")
        
        return profile_content

    def answer_question(self, document_chunks: List[str], question: str) -> str:
        """Answer a user question based on the document context."""
        context = "\n\n".join(document_chunks)
        
        # Identify the types of documents based on content - same as in generate_profile
        assessment_types = []
        
        # Check for psychological assessment content
        assessment_terms = ["psychological assessment", "psych eval", "mental status", "diagnosis", "dsm", "icd", "symptoms"]
        has_assessment = any(term in context.lower() for term in assessment_terms)
        if has_assessment:
            assessment_types.append("Psychological Assessment")
            
        # Check for treatment notes content
        treatment_terms = ["treatment notes", "therapy notes", "session notes", "progress notes"]
        has_treatment = any(term in context.lower() for term in treatment_terms)
        if has_treatment:
            assessment_types.append("Treatment Notes")
            
        # Check for medical history content
        medical_terms = ["medical history", "medication", "health history", "physical exam", "vitals"]
        has_medical = any(term in context.lower() for term in medical_terms)
        if has_medical:
            assessment_types.append("Medical History")
        
        # Check for clinical interview
        interview_terms = ["clinical interview", "intake", "initial assessment", "client report"]
        has_interview = any(term in context.lower() for term in interview_terms)
        if has_interview:
            assessment_types.append("Clinical Interview")
        
        # Check for standardized tests
        test_terms = ["mmpi", "wais", "wisc", "beck", "hamilton", "gaf", "phq", "gad"]
        has_tests = any(term in context.lower() for term in test_terms)
        if has_tests:
            assessment_types.append("Standardized Tests")
            
        # Check for personality assessment content
        hogan_terms = ["hogan", "hpi", "hds", "mvpi", "personality inventory"]
        has_hogan = any(term in context.lower() for term in hogan_terms)
        if has_hogan:
            assessment_types.append("Personality Assessment")
            
        # Combine detected document types
        detected_doc_types = ", ".join(assessment_types) if assessment_types else "Submitted Documents"
        
        # Add the same citation guidance as in generate_profile
        citation_guidance = """
EXTREMELY IMPORTANT GUIDANCE ON SOURCES AND CITATIONS:

1. When citing sources, DO NOT refer to them by their file type (e.g., 'PDF', 'DOCX'). Instead, identify them by their content type:
   - Refer to psychological assessments as 'Psychological Assessment' 
   - Refer to therapy documentation as 'Treatment Notes'
   - Refer to medical information as 'Medical History'
   - Refer to personality measures as 'Personality Assessment'
   - For other documents, identify them by their purpose (e.g., 'Clinical Interview', 'Standardized Tests')

2. For EVERY significant claim or insight in your analysis, include a brief in-text citation showing the source, like this: 
   '... exhibits anxiety symptoms (Psychological Assessment).' or '... has responded well to CBT techniques (Treatment Notes).'

3. Do not make claims that cannot be directly supported by the provided documents. If you're unsure about a claim, clearly indicate this.

4. At the end of your response, include a "References" section that lists all the source documents you cited.

5. Every paragraph should include at least one specific citation to a source document.

6. DO NOT HALLUCINATE OR INVENT SOURCES. Only use the document types that have been detected in the uploaded materials:
   {detected_doc_types}
"""

        prompt = f"""Based on the following patient documents, answer this special question from the mental health practitioner:

{citation_guidance}

{context}

Question: {question}

Please provide a detailed, evidence-based answer, providing specific in-text citations for each claim (e.g., "exhibits anxiety symptoms (Psychological Assessment)").

End your response with a "References" section that lists all the documents you cited.

Remember: Only make claims that are directly supported by the documents. Include parenthetical citations for each major claim."""

        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=4000
        )
        return response.choices[0].message.content


