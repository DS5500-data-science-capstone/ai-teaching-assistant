# AI Teaching Assistant for Canvas LMS

An AI-powered teaching assistant that integrates with Canvas Learning Management System to provide automated learning analytics, at-risk student detection, and personalized instructional support.

##  Project Overview

This capstone project develops an intelligent system that helps faculty enhance student learning through data-driven insights and AI-powered support. The system analyzes Canvas course data to identify struggling students, detect learning gaps, and provide actionable recommendations for timely intervention.

### Key Features (Planned)

-  **At-Risk Student Detection**: Automated identification of students needing intervention
-  **Learning Analytics Dashboard**: Visual insights into class performance and trends
-  **AI-Powered Communication**: Generate personalized check-in emails for students
-  **RAG-Based Tutoring**: Answer student questions using course materials
-  **Assignment Analytics**: Identify difficult topics and common misconceptions

## Team Members

- **Priyadharshan Sengutuvan** - System Architecture, RAG Pipeline, Canvas Integration
- **Anjana Deivasigamani** - Data Analysis, Evaluation Metrics, Risk Detection
- **Raghu Ram Baskaran** - Dashboard Development, Monitoring, Documentation

##  Architecture
```
Faculty Dashboard (Streamlit)
         ↓
    FastAPI Backend
         ↓
    ┌────┴────┐
Canvas API  PostgreSQL  ChromaDB  OpenAI API
```

##  Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: PostgreSQL
- **Vector DB**: ChromaDB
- **AI/ML**: OpenAI GPT-4, LangChain
- **LMS Integration**: Canvas API (canvasapi)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+
- Canvas LMS API access token
- OpenAI API key

## Setup Instructions

### 1. Clone the Repository

\`\`\`bash
git clone https://github.com/[na4sh11]/ai-teaching-assistant-canvas.git
cd ai-teaching-assistant-canvas
\`\`\`

### 2. Create Virtual Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
\`\`\`

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Configure Environment Variables

\`\`\`bash
# Copy example env file
cp .env.example .env

# Edit .env with your actual credentials
nano .env
\`\`\`

### 5. Initialize Database

\`\`\`bash
python scripts/setup_db.py
\`\`\`

### 6. Run the Application

\`\`\`bash
# Start the dashboard
streamlit run src/dashboard/app.py
\`\`\`

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [API Reference](docs/api-reference.md) *(Coming soon)*

## Running Tests

\`\`\`bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
\`\`\`

## Project Status

**Current Iteration: Month 1 - Foundation**

- [x] Repository setup
- [x] Project structure established
- [ ] Canvas API integration
- [ ] Database schema design
- [ ] Basic dashboard UI
- [ ] At-risk detection algorithm




**Last Updated:** January 2025  
**Academic Term:** Spring 2025  
**Course:** MSDS Capstone Project
