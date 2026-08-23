"""UETM Assistant — RAG chatbot for UET Mardan.

Loads the local vector index built by ingest.py, retrieves the most relevant
chunks for a question, and answers strictly from that context using OpenAI.
Also provides a catalog of application forms (built from the prospectus) so
students asking about forms get told exactly which form applies to them.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = 6
MAX_HISTORY = 6

# Built from the official UET Mardan prospectus (2022-23) during dataset analysis.
FORMS_CATALOG = [
    {
        "id": "admission",
        "name": "Admission Application Form (BSc / BS)",
        "for": "New applicants applying to any BSc Engineering or BS program "
               "(Open Merit, Self-Finance, or Quota seats).",
        "where": "Submit online to the Directorate of Admissions, UET Mardan "
                 "before the advertised last date (www.uetmardan.edu.pk).",
        "fee": "As per admission fee table (admission fee + first semester dues)",
        "notes": "Attach attested copies of SSC & HSSC/Intermediate DMCs, "
                 "equivalence certificate (if applicable), Hafiz-e-Quran "
                 "certificate (if claiming), and other documents listed in "
                 "prospectus section 11.7.",
        "url": "https://uetmardan.edu.pk/ugadmissions/",
    },
    {
        "id": "correction",
        "name": "Correction / Rectification Form",
        "for": "Applicants who need to correct/rectify data in an already "
               "submitted admission application.",
        "where": "Available at the Directorate of Admission, UET Mardan.",
        "fee": "Rs. 200/-",
        "notes": "Attach a copy of the previously submitted application form "
                 "with the Correction/Rectification Form.",
    },
    {
        "id": "migration",
        "name": "Migration Application (to UET Mardan)",
        "for": "Students of other universities who want to migrate into UET "
               "Mardan (eligible only in certain semesters).",
        "where": "Submit to the Dean, Faculty of Engineering & Computing "
                 "within 15 days of the commencement of the semester.",
        "fee": "Rs. 200,000/- migration fee (payable to Treasurer, UET Mardan)",
        "notes": "Requires NOC from the previous institution and an up-to-date "
                 "official transcript; minimum CGPA 2.00 for completed courses.",
    },
    {
        "id": "hostel",
        "name": "Hostel Admission Form",
        "for": "Students on regular rolls of the university seeking hostel "
               "accommodation.",
        "where": "Submit to the hostel dealing office / Provost office before "
                 "the announced last date.",
        "fee": "Hostel dues as per hostel regulations",
        "notes": "Attach five passport-size photographs duly attested by the "
                 "Head of the department. Students on the non-subsidized "
                 "(self-finance) scheme are not entitled to boarding.",
    },
    {
        "id": "semester-registration",
        "name": "Semester Registration Form",
        "for": "All students registering for a new semester (every semester, "
               "in order).",
        "where": "Get it checked and signed by the Batch Advisor, then deposit "
                 "in the department office before the deadline.",
        "fee": "Semester dues as announced",
        "notes": "Registration for successive semesters must be in order; "
                 "late registration attracts the late fee.",
    },
    {
        "id": "freeze",
        "name": "Semester Freeze Application",
        "for": "Students who want to freeze (interrupt) a semester.",
        "where": "Submit application to the department/controller's office "
                 "within the specified time.",
        "fee": "Rs. 5,000/- semester freezing fee",
        "notes": "Semester freezing is subject to rules in the Academic & "
                 "Examination Regulations.",
    },
    {
        "id": "rechecking",
        "name": "Rechecking of Scripts (per paper)",
        "for": "Students who want their exam paper rechecked.",
        "where": "Apply through the Examination Section (Controller of "
                 "Examinations).",
        "fee": "Rs. 700/- per paper",
        "notes": "Paid in advance; fee is non-refundable.",
    },
    {
        "id": "transcript",
        "name": "Transcript / Degree Certificate Application",
        "for": "Students requesting transcripts, provisional certificates, or "
               "their degree certificate.",
        "where": "Controller of Examinations — apply using the official "
                 "downloadable form from the university website.",
        "fee": "Provisional certificate Rs. 1,000; duplicate provisional "
               "Rs. 800; duplicate degree Rs. 2,500; verification Rs. 1,500 "
               "per set; convocation Rs. 1,000",
        "notes": "Fees per Table (4) of the prospectus. There are two separate "
                 "forms: one for the degree and one for transcript & "
                 "provisional certificate.",
        "url": "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Application_form_for_Transcript_v2.pdf",
    },
    {
        "id": "degree-form",
        "name": "Application Form for Degree",
        "for": "Students applying for issuance of their degree certificate.",
        "where": "Available online from the university downloads section; "
                 "submit to the Controller of Examinations.",
        "fee": "As per prospectus Table (4).",
        "notes": "Official download: Assets/Downloads - Application form for Degree.",
        "url": "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Application_form_for_Degree_v2.pdf",
    },
    {
        "id": "grievance",
        "name": "QEC Grievance / Complaint Form",
        "for": "Students with grievances against any department, section, "
               "center, directorate, or employee of the university.",
        "where": "Available at the Directorate of QEC or the download section "
                 "of the university website (www.uetmardan.edu.pk).",
        "fee": "None",
        "notes": "Submit the complaint on the prescribed form to the Director "
                 "Quality Enhancement Cell (QEC).",
    },
]

# Topic keywords so the assistant can focus retrieval and pick the right form.
INTENT_KEYWORDS = {
    "admission": ["admission", "apply", "apply for", "eligib", "merit", "entry test", "entrance"],
    "fee": ["fee", "fees", "cost", "charges", "amount", "dues", "tuition", "scholarship", "free-ship", "concession"],
    "form": ["form", "application form", "apply form", "download form"],
    "download": ["download", "downloads", "file", "pdf", "document", "proforma", "prospectus file"],
    "hostel": ["hostel", "boarding", "accommodation", "mess", "warden"],
    "academics": ["course", "semester", "credit", "syllabus", "scheme", "subject", "degree", "program", "curriculum", "research", "lab"],
    "grades": ["cgpa", "gpa", "grade", "probation", "freeze", "recheck", "reappear", "result", "dismiss"],
    "exam": ["exam", "paper", "test", "assessment", "mark", "transcript", "migration"],
    "contact": ["contact", "email", "phone", "address", "office", "directorate", "registrar", "admissions@", "extension"],
}

# All downloadable files from the official UET Mardan website:
# https://www.uetmardan.edu.pk/uetm/Download
DOWNLOADS_CATALOG = [
    # Application Forms
    ("Application Forms", "Job Application Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/21_Job-Application-Form.pdf"),
    ("Application Forms", "Income Proforma", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Income%20Proforma.pdf"),
    ("Application Forms", "GP Fund Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/GP-Fund-Form-UETM.pdf"),
    ("Application Forms", "Service Card Application Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Service-card-application-form.pdf"),
    ("Application Forms", "Faculty Staff Hostel Accommodation Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Faculty-Staff-Accommodation-Application-Form.pdf"),
    ("Application Forms", "School Subsidy Proforma (children of UETM employees)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/141019_School_subsidy_form-23.pdf"),
    ("Application Forms", "Student Official Email Proforma", "https://www.uetmardan.edu.pk/uetm/assets/files/news_files/Student_official_Email_Proforma.pdf"),
    # KP Universities Acts
    ("KP Universities Acts", "Khyber Pakhtunkhwa Universities Act 2012", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/260419-THE_KHYBER_PAKHTUNKHWA_UNIVERSITIES_ACT_2012.pdf"),
    ("KP Universities Acts", "Universities Amendment Act 2018", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/260419-Universities-Amendment-Act20180001.pdf"),
    ("KP Universities Acts", "Gazette Notification of KP Universities Act 2024", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Gazzate%20notification%20of%20KP%20universities%20act%202024_260729_193823.pdf"),
    # UET Mardan Statutes
    ("UET Mardan Statutes", "UET Mardan Statutes 2022", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads//uetm/assets/files/downloads/scan-statutes-2023.pdf"),
    ("UET Mardan Statutes", "Amendments/Anomalies (2025) in UET Mardan Statutes 2022", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/anomly-amendment-statutes.pdf"),
    ("UET Mardan Statutes", "UET Mardan Statutes 2022 (Amended 2025)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/uetm-statutes-amended-2022.pdf"),
    # Budget
    ("Budget", "Approved Budget Estimate 2023-24", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/budget2023-2024.jpeg"),
    ("Budget", "Approved Budget Estimate 2024-25", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Approved-Budget-estimate-2024-25.jpeg"),
    ("Budget", "Approved Budget Summary 2025-26", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Approved-Bdget-Summary-25-26.pdf"),
    ("Budget", "Filled and Vacant Position Summary", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Filled-and-Vacant-Position--summary.jpeg"),
    # Rules
    ("Rules", "UET Mardan Administrative Rules, Regulations, Policies & Processes 2018", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/WhatsApp%20Image%202026-07-30%20at%2012.44.27%20AM.jpeg"),
    ("Rules", "UET Mardan Financial Rules 2018", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/WhatsApp%20Image%202026-07-30%20at%2012.48.30%20AM.jpeg"),
    ("Rules", "Employees Residency Rules 2022", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Employees-Residency-Rules-2022.pdf"),
    ("Rules", "Medical Rules for Reimbursement 2023", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/medical-rules-2023.pdf"),
    ("Rules", "SoPs and Rules for Short Courses 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/short-courses-sops-2026.pdf"),
    # Policies
    ("Policies", "Transition Education, HEC-Foreign-Collaboration Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Transition%20education,%20HEC-Foreign-Collaboratin-Policy.pdf"),
    ("Policies", "Students Disability Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Policy%20for%20Students%20with%20Disabilities%202021-%20Amended.pdf"),
    ("Policies", "Framework for HEC Best University Teacher Award", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Framework%20for%20HEC%20Best%20University%20Teacher%20award.pdf"),
    ("Policies", "Best Researcher Award — UET Mardan", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Best%20Researcher%20Award-UET%20Mardan-Final%20Version.pdf"),
    ("Policies", "Faculty Workload Distribution Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Faculty%20Workload%20Distribution%20Policy%20-%20Updated.pdf"),
    ("Policies", "Conflict of Interest Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Conflict%20of%20Interest%20Policy-Revised%20Clean%20Version.pdf"),
    ("Policies", "Entrepreneurship Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/UET_Mardan_Entrepreneurship_Policy%5B1%5D.pdf"),
    ("Policies", "International Students Credit Transfer Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/UET_Mardan_International_Students_and_Credit_Transfer_Policy.pdf"),
    ("Policies", "Handover Takeover Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/UETM_Handover_Takeover_Policy.pdf"),
    ("Policies", "Whistle Blower Protection Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/WHISTLEBLOWER%20PROTECTION%20POLICY.pdf"),
    ("Policies", "Intellectual Property Rights Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Intellectual%20Property%20Rights%20(IPR)%20policy.pdf"),
    ("Policies", "Plagiarism Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Plagiarism-Policy.pdf"),
    ("Policies", "Consultancy Policy", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/consultancy-policy-21-Synd.pdf"),
    ("Policies", "Job Descriptions", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Job_Desc-UETM.pdf"),
    ("Policies", "Alumni Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Alumni%20Policy%202026.pdf"),
    ("Policies", "SCALE Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/SCALE%20Policy%202026.pdf"),
    ("Policies", "EHS Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/EHS%20Policy%202026.pdf"),
    ("Policies", "ORIC Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/ORIC%20Policy%202026.pdf"),
    ("Policies", "Visiting Faculty Engagement Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Visiting%20Faculty%20Engagement%20Policy%202026.pdf"),
    ("Policies", "PG Courses Renumeration Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG%20courses%20renumeration%20policy%202026.pdf"),
    ("Policies", "PG Admissions SOPs", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG%20Admissions%20SOPs.pdf"),
    ("Policies", "PG Test Prep Conduct & Evaluation Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG%20Test%20Prep%20Conduct%20and%20evaluation%20Policy%202026.pdf"),
    ("Policies", "External Evaluation of PhD Dissertation Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/External%20Evaluation%20of%20PhD%20Dissertation%20Policy%202026.pdf"),
    ("Policies", "Harassment Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/HARASSMENT%20POLICY_2026.pdf"),
    ("Policies", "PG Admission Cancellation Policy 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/CamScanner%2009-07-2026%201618_26_260730_145021.pdf"),
    ("Policies", "PG Duration Extension Guidelines 2026", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/CamScanner%2009-07-2026%201618_25_260730_145315.pdf"),
    # Seniority
    ("Seniority", "House Allotment Seniority List 2023", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/house-allotment-list-2023.pdf"),
    # Academic Calendars
    ("Academic Calendars", "UG Academic Calendar 2025-26", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/WhatsApp%20Image%202025-07-08%20at%208.58.48%20PM.jpeg"),
    ("Academic Calendars", "Summer Academic Calendar 2025", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/summer-2025.jpeg"),
    ("Academic Calendars", "PG Academic Calendar 2024-25", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/WhatsApp%20Image%202025-08-29%20at%202.33.43%20PM%20(1).jpeg"),
    # Grievance Notifications
    ("Grievance Notifications", "Grievance Redressal Committee for TTS Faculty", "https://www.uetmardan.edu.pk/uetm/assets/files/news_files/WhatsApp%20Image%202024-08-01%20at%209.41.56%20AM.jpeg"),
    ("Grievance Notifications", "Grievance Management Cell", "https://www.uetmardan.edu.pk/uetm/assets/files/news_files/WhatsApp%20Image%202024-08-01%20at%209.41.55%20AM.jpeg"),
    # Transcript & Degree Forms
    ("Transcript & Degree Forms", "Application Form for Degree", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Application_form_for_Degree_v2.pdf"),
    ("Transcript & Degree Forms", "Application Form for Transcript & Provisional Certificate", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Application_form_for_Transcript_v2.pdf"),
    # Newsletters
    ("Newsletters", "Newsletter 2021", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Newsletter%20UET2.pdf"),
    ("Newsletters", "Newsletter 2025", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/UETM-newsletter-12.02.2026.pdf"),
    # Student Forms
    ("Student Forms", "Transportation Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/11122020_UETM_Transportation_Form.pdf"),
    ("Student Forms", "BSc Registration Form (Regular Courses)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/260419-BSc-Registration-Form-Regular-courses.pdf"),
    ("Student Forms", "Re-Registration Form", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/260419-Re-registration-form.pdf"),
    ("Student Forms", "Re-Registration Form for Summer Semester 2024", "https://uetmardan.edu.pk/uetm/assets/files/downloads/Summer_2024_Re-registration_form.pdf"),
    ("Student Forms", "FYP Manual", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/fyp-manual-11.2024.pdf"),
    ("Student Forms", "Information of Batch Advisors", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Information_of_Batch_Advisors_of_Teaching_Departments_of_UET_Mardan.pdf"),
    ("Student Forms", "Online Assessment's Grievance Form (Form B)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Online_Assessment%E2%80%99s_Grievance_Form_%28Form%20B%29_Spring_2020.pdf"),
    ("Student Forms", "Course(s) Withdrawal Request Form (Form A)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Course%28s%29_Withdrawal_Request_Form_%28Form%20A%29_Spring_2020.pdf"),
    ("Student Forms", "Checklist & Instructions (Spring/Summer 2020)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Checklist_for_Students.pdf"),
    ("Student Forms", "Grading Criteria & SOPs for Assessment (Spring 2020)", "https://www.uetmardan.edu.pk/assets/files/downloads/SoPs-Examinations%20and%20Assessment-(23rd%20June%202020).pdf"),
    # Online Classes Proformas
    ("Online Classes Proformas", "Online Course Readiness Performa", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Online_Course_Readiness_Performa.docx"),
    ("Online Classes Proformas", "Online Course Evaluation Performa", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Online_Course_Evaluation_Performa.docx"),
    ("Online Classes Proformas", "Online Course Approval Performa", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Online_Course_Approval_Performa.docx"),
    ("Online Classes Proformas", "Online Lab Course Readiness Performa", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/Online_Lab_Course_Readiness_Performa.docx"),
    ("Online Classes Proformas", "Online Course Quality Monitoring Performa", "https://www.uetmardan.edu.pk/assets/files/downloads/online_course_quality_monitoring_performa.docx"),
    # Postgraduate Forms
    ("Postgraduate Forms", "PG Form-1 (Student Registration Form)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG-Form-1%20_%20(Student%20Registration%20Form).docx"),
    ("Postgraduate Forms", "PG Form-6 (Issuance of Notification for Award of Degree)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG%20Form-6_%20(Issuance%20of%20Notification%20for%20Award%20of%20Degree).docx"),
    ("Postgraduate Forms", "PG Form-9 (Paper Publication Certificate)", "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/PG%20Form-9_%20(Paper%20Publication%20Certificate).docx"),
]


def intent_for(question: str) -> list[str]:
    q = question.lower()
    return [name for name, words in INTENT_KEYWORDS.items() if any(w in q for w in words)]


def load_index():
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(INDEX_DIR / "vectors.npy")
    return chunks, vectors


def retrieve(question: str, client: OpenAI, chunks: list, vectors: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    qv = np.array(
        client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding,
        dtype=np.float32,
    )
    scores = vectors @ qv  # cosine similarity (vectors are L2-normalized)
    top = np.argsort(scores)[-top_k:][::-1]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in top
        if scores[i] > 0.15
    ]


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source {i}]\n{r['text']}")
    return "\n\n".join(parts)


def build_forms_context() -> str:
    lines = []
    for f in FORMS_CATALOG:
        entry = (
            f"- {f['name']} (for: {f['for']})\n"
            f"  Where: {f['where']}\n"
            f"  Fee: {f['fee']}\n"
            f"  Notes: {f['notes']}"
        )
        if f.get("url"):
            entry += f"\n  Download: {f['url']}"
        lines.append(entry)
    return "\n".join(lines)


def build_downloads_context() -> str:
    """Render the full downloadable-files catalog grouped by category."""
    groups: dict[str, list[str]] = {}
    for category, title, url in DOWNLOADS_CATALOG:
        groups.setdefault(category, []).append(f"  - {title}: {url}")
    return "\n".join(
        f"{category}\n" + "\n".join(items)
        for category, items in groups.items()
    )


SYSTEM_PROMPT = """You are "UETM Assistant," an advanced, intelligent AI agent for the University of Engineering and Technology (UET) Mardan. Your job is to help current students, applicants, and faculty with accurate university information.

### 1. KNOWLEDGE SOURCE & BOUNDARIES
- Primary Source: You must rely STRICTLY on the provided context retrieved from the UET Mardan prospectus, handbooks, and official documents, plus the official forms catalog and downloads catalog supplied below when they are included.
- Fallback Rule: If a user asks a specific question (e.g., "What is the fee for semester 3?", "When is the paper rechecking deadline?") and the retrieved context DOES NOT contain the explicit answer, DO NOT invent facts.
- Response when uncertain: Say, "I don't have that specific official detail in my current database. Please contact the UET Mardan Directorate of Admissions at admissions@uetmardan.edu.pk or visit the Registrar's office."

### 2. FORMS & DOWNLOADS GUIDANCE
- When the user asks about any form (admission, correction, migration, hostel, semester registration, freeze, rechecking, transcript, degree, grievance), use the FORMS CATALOG below to tell them:
  * the exact form name,
  * who it is for,
  * where to get it and submit it,
  * any fee, and
  * required documents.
- When the user asks for any file or document (form, policy, act, statute, budget, calendar, newsletter, proforma), use the DOWNLOADS CATALOG below and give them the exact official download link.
- Never invent forms or download links that are not in the catalogs.

### 3. LANGUAGE & TONE
- Tone: Professional, polite, concise, and structured.
- Multilingual Support:
  * Respond in the primary language used by the student (English, Roman Urdu, or Urdu).
  * If the student asks in Roman Urdu (e.g., "CS ki fee kitni hai?"), respond in clear, grammatically correct English or easy Roman Urdu, prioritizing clear formatting (tables/bullet points).

### 4. DEPARTMENTS & ACADEMICS FRAMEWORK
Recognize UET Mardan's key academic domains:
- Departments: Computer Software Engineering, Computer Science, Telecommunication Engineering, Electrical Engineering, Mechanical Engineering, Civil Engineering, and Center of Artificial Intelligence.
- Categories: Admission eligibility (FSc Pre-Engineering/ICS/DAE), grading rules (CGPA, probation, semester freeze), fee structures (Open Merit, Self-Sustain, Self-Finance), and hostel rules.

### 5. OUTPUT FORMATTING
- Summarize complex policies into bullet points or Markdown tables.
- For fee queries, always break down: Admission Fee + Semester Fee + Security (if hosteled).
- Always include relevant contact info (e.g., email or office extension) when directing users to an administrative desk.

### 6. SAFETY & GUARDRAILS
- Never answer non-academic, harmful, or political questions.
- Maintain neutrality and strict adherence to UET Mardan policies.
"""


app = Flask(__name__, static_folder="static", static_url_path="")
_client = None
_chunks = None
_vectors = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # raises if OPENAI_API_KEY is missing
    return _client


def get_index():
    global _chunks, _vectors
    if _chunks is None or _vectors is None:
        _chunks, _vectors = load_index()  # raises if index/ is missing
    return _chunks, _vectors


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/forms")
def forms():
    return jsonify({"forms": FORMS_CATALOG})


@app.get("/api/downloads")
def downloads():
    grouped: dict[str, list[dict]] = {}
    for category, title, url in DOWNLOADS_CATALOG:
        grouped.setdefault(category, []).append({"title": title, "url": url})
    return jsonify({"categories": grouped})


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    question = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "Message is required."}), 400

    try:
        client = get_client()
        chunks, vectors = get_index()
    except Exception as e:
        print(f"Setup error: {e}")
        return jsonify({
            "error": (
                "The assistant is not configured yet. Add your OPENAI_API_KEY to .env, "
                "then run `python ingest.py` to build the index, and restart the app."
            )
        }), 503

    intents = intent_for(question)

    results = retrieve(question, client, chunks, vectors)
    if not results and "form" not in intents and "download" not in intents:
        return jsonify({
            "reply": "I don't have that specific official detail in my current database. "
                     "Please contact the UET Mardan Directorate of Admissions at "
                     "admissions@uetmardan.edu.pk or visit the Registrar's office.",
        })

    user_content = (
        f"Retrieved context from the UET Mardan official documents:\n\n"
        f"{build_context(results)}\n\n"
    )
    if "form" in intents:
        user_content += (
            "FORMS CATALOG (official, from the prospectus):\n\n"
            f"{build_forms_context()}\n\n"
        )
    if "download" in intents:
        user_content += (
            "DOWNLOADS CATALOG (official files available on the university website):\n\n"
            f"{build_downloads_context()}\n\n"
        )
    user_content += (
        f"Question: {question}\n\n"
        "Answer using ONLY the retrieved context above and the forms/downloads catalogs "
        "if provided. If the context does not contain the explicit answer, use the "
        "fallback response instead of inventing facts."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": user_content})

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return jsonify({
        "reply": completion.choices[0].message.content,
        "intents": intents,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
