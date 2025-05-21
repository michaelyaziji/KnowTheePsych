# KnowThee.AI - Psychological Assessment Tool for Mental Health Practitioners

An AI-powered psychological assessment tool for mental health practitioners that generates personalized psychology profiles for patients from various document inputs.

## Features
- Upload and process PDF and DOCX files (psychological assessments, medical history, treatment notes)
- AI-powered comprehensive psychological profile generation
- HIPAA-compliant privacy-first design
- Export profiles to PowerPoint for treatment planning

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run app.py
```

## Privacy
This application is designed with strict privacy and HIPAA compliance in mind:
- No long-term storage of PHI (Protected Health Information) without explicit permission
- Data is processed locally where possible
- Temporary storage only for the duration of the session
- No external data sharing beyond what's needed for AI processing

## License
Proprietary - All rights reserved 