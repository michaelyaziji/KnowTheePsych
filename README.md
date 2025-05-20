# KnowThee.AI - Psychology Assessment Tool

An AI-powered psychology assessment and development tool that generates personalized psychology profiles from various document inputs.

## Features
- Upload and process PDF and DOCX files (CVs, 360s, psychometric assessments)
- AI-powered psychology profile generation
- Privacy-first design
- Export profiles to PDF

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
This application is designed with privacy in mind:
- No long-term storage of PII without explicit permission
- Data is processed locally where possible
- Temporary storage only for the duration of the session

## License
Proprietary - All rights reserved 