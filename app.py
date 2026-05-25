import streamlit as st
import joblib
import pdfplumber
import matplotlib.pyplot as plt
import io
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#for pdf generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

model = joblib.load('model.pkl')
encode = joblib.load('label.pkl')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.title('AI Resume Analyzer')
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True)
pdf = st.file_uploader("Upload Your Resume", type=['pdf'])
jd = st.text_input("Enter Job Description")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
skills_db = {
    # ===================== IT / SOFTWARE =====================
    "it": {

        "Programming Languages": [
            "python", "java", "c++", "c",
            "javascript", "typescript"
        ],

        "Frontend Development": [
            "html", "css", "react",
            "angular", "vue"
        ],

        "Backend Development": [
            "node.js", "express.js",
            "django", "flask", "fastapi",
            "spring boot"
        ],

        "API & Architecture": [
            "rest api", "graphql",
            "microservices",
            "system design"
        ],

        "Database": [
            "sql", "mysql",
            "postgresql", "mongodb",
            "redis"
        ],

        "Cloud & DevOps": [
            "docker", "kubernetes",
            "aws", "azure", "gcp",
            "linux", "git", "github"
        ],

        "Core Concepts": [
            "data structures",
            "algorithms",
            "api development",
            "backend development",
            "frontend development"
        ]
    },

    # ===================== DATA SCIENCE / AI =====================

    "data_science": {

        "Programming": [
            "python", "r", "sql"
        ],

        "Machine Learning": [
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision"
        ],

        "Libraries": [
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "pytorch",
            "keras"
        ],

        "Analytics": [
            "data analysis",
            "data visualization",
            "statistics",
            "probability",
            "hypothesis testing"
        ],

        "Visualization Tools": [
            "power bi",
            "tableau",
            "excel"
        ],

        "Big Data": [
            "big data",
            "hadoop",
            "spark"
        ],

        "Advanced Concepts": [
            "feature engineering",
            "model deployment"
        ]
    },

    # ===================== FINANCE =====================

    "finance": {

        "Accounting": [
            "financial analysis",
            "accounting",
            "bookkeeping",
            "taxation",
            "audit"
        ],

        "Banking": [
            "banking operations",
            "credit analysis",
            "loan processing",
            "insurance underwriting"
        ],

        "Investment": [
            "investment analysis",
            "portfolio management",
            "asset management",
            "equity research",
            "trading",
            "derivatives"
        ],

        "Financial Planning": [
            "financial modeling",
            "valuation",
            "budgeting",
            "forecasting",
            "financial reporting",
            "excel modeling",
            "risk management"
        ]
    },

    # ===================== MANAGEMENT / BUSINESS =====================

    "management": {

        "Project Management": [
            "project management",
            "agile",
            "scrum",
            "kanban",
            "waterfall"
        ],

        "Business Skills": [
            "business analysis",
            "business development",
            "strategic planning",
            "operations management"
        ],

        "Leadership": [
            "leadership",
            "team management",
            "decision making",
            "problem solving",
            "time management"
        ],

        "Communication": [
            "communication",
            "crm",
            "stakeholder management"
        ]
    },

    # ===================== MARKETING =====================

    "marketing": {

        "Digital Marketing": [
            "digital marketing",
            "seo",
            "sem",
            "social media marketing"
        ],

        "Advertising": [
            "google ads",
            "facebook ads",
            "affiliate marketing"
        ],

        "Content": [
            "content marketing",
            "copywriting",
            "email marketing"
        ],

        "Business Growth": [
            "brand management",
            "market research",
            "lead generation",
            "conversion optimization"
        ]
    },

    # ===================== HR =====================

    "hr": {

        "Recruitment": [
            "recruitment",
            "talent acquisition",
            "onboarding"
        ],

        "Employee Management": [
            "employee relations",
            "performance management",
            "training and development"
        ],

        "HR Operations": [
            "payroll management",
            "hr policies",
            "hr analytics"
        ]
    },

    # ===================== SALES =====================

    "sales": {

        "Sales Types": [
            "b2b sales",
            "b2c sales"
        ],

        "Lead Management": [
            "lead generation",
            "cold calling"
        ],

        "Customer Handling": [
            "negotiation",
            "customer relationship management",
            "account management"
        ],

        "CRM Tools": [
            "crm",
            "salesforce",
            "sales strategy"
        ]
    },

    # ===================== OPERATIONS =====================

    "operations": {

        "Supply Chain": [
            "supply chain management",
            "logistics",
            "inventory management",
            "procurement"
        ],

        "Optimization": [
            "process optimization",
            "quality assurance",
            "six sigma",
            "lean management"
        ]
    },

    # ===================== DESIGN =====================

    "design": {

        "UI/UX": [
            "ui design",
            "ux design",
            "wireframing",
            "prototyping",
            "interaction design"
        ],

        "Graphic Design": [
            "graphic design",
            "adobe photoshop",
            "illustrator"
        ],

        "Design Tools": [
            "figma",
            "sketch"
        ],

        "Research": [
            "user research"
        ]
    },

    # ===================== CYBERSECURITY =====================

    "cybersecurity": {

        "Security Testing": [
            "ethical hacking",
            "penetration testing",
            "vulnerability assessment"
        ],

        "Network Security": [
            "network security",
            "firewall management",
            "information security"
        ],

        "Security Concepts": [
            "cryptography",
            "risk assessment",
            "security auditing"
        ]
    },

    # ===================== HEALTHCARE =====================

    "healthcare": {

        "Core Medical Skills": [
            "patient care",
            "medical diagnosis",
            "clinical assessment",
            "treatment planning",
            "disease management"
        ],

        "Medical Systems": [
            "medical documentation",
            "electronic health records",
            "ehr",
            "emr"
        ],

        "Nursing": [
            "nursing",
            "critical care",
            "icu care",
            "emergency care",
            "patient monitoring",
            "vital signs monitoring",
            "medication administration",
            "wound care",
            "infection control"
        ],

        "Pharmacy": [
            "pharmacy",
            "clinical pharmacy",
            "drug dispensing",
            "pharmacology",
            "prescription management",
            "medication therapy management"
        ],

        "Clinical Research": [
            "clinical research",
            "clinical trials",
            "medical research",
            "biostatistics",
            "epidemiology",
            "research methodology",
            "data collection"
        ],

        "Medical Coding": [
            "medical coding",
            "medical billing",
            "icd coding",
            "cpt coding",
            "health insurance",
            "claims processing"
        ],

        "Diagnostics": [
            "radiology",
            "medical imaging",
            "mri",
            "ct scan",
            "x ray",
            "ultrasound",
            "pathology",
            "laboratory testing",
            "diagnostic testing"
        ],

        "Healthcare Administration": [
            "healthcare management",
            "hospital administration",
            "healthcare operations",
            "patient scheduling",
            "healthcare compliance",
            "hipaa compliance"
        ],

        "Biotechnology": [
            "biotechnology",
            "genetics",
            "molecular biology",
            "microbiology",
            "biochemistry"
        ],

        "Public Health": [
            "public health",
            "community health",
            "health education",
            "vaccination",
            "disease prevention"
        ],

        "Therapy": [
            "physiotherapy",
            "occupational therapy",
            "speech therapy",
            "rehabilitation"
        ],

        "Soft Skills": [
            "communication",
            "empathy",
            "team collaboration",
            "problem solving",
            "time management"
        ],

        "Healthcare Tools": [
            "excel",
            "data analysis",
            "healthcare analytics",
            "telemedicine",
            "patient management systems"
        ]
    },

    # ===================== ENGINEERING =====================

    "engineering": {

        "Core Engineering": [
            "mechanical design",
            "civil engineering",
            "electrical engineering",
            "electronics"
        ],

        "Design Tools": [
            "cad",
            "autocad",
            "solidworks"
        ],

        "Project Management": [
            "project estimation",
            "construction management",
            "thermodynamics"
        ]
    },

    # ===================== LEGAL =====================

    "legal": {

        "Law": [
            "contract law",
            "corporate law",
            "litigation",
            "arbitration"
        ],

        "Legal Operations": [
            "legal research",
            "compliance",
            "regulatory affairs",
            "legal drafting"
        ],

        "Specialization": [
            "intellectual property"
        ]
    },

    # ===================== EDUCATION =====================

    "education": {

        "Teaching": [
            "teaching",
            "classroom management",
            "student assessment"
        ],

        "Curriculum": [
            "curriculum development",
            "instructional design",
            "e-learning"
        ],

        "Research & Training": [
            "academic research",
            "training delivery"
        ]
    }
}
role_mapping = {

    # ===================== IT / SOFTWARE =====================

    "software developer": "it",
    "software engineer": "it",
    "web developer": "it",
    "frontend developer": "it",
    "backend developer": "it",
    "full stack developer": "it",
    "fullstack developer": "it",
    "python developer": "it",
    "java developer": "it",
    "react developer": "it",
    "node js developer": "it",
    "django developer": "it",
    "devops engineer": "it",
    "cloud engineer": "it",
    "api developer": "it",
    "it":"it",
    "information technology":"it",

    # ===================== DATA SCIENCE / AI =====================

    "data scientist": "data_science",
    "data analyst": "data_science",
    "machine learning engineer": "data_science",
    "ml engineer": "data_science",
    "ai engineer": "data_science",
    "deep learning engineer": "data_science",
    "nlp engineer": "data_science",
    "computer vision engineer": "data_science",
    "business intelligence analyst": "data_science",
    "bi analyst": "data_science",

    # ===================== FINANCE =====================

    "banker": "finance",
    "banking": "finance",
    "bank manager": "finance",
    "financial analyst": "finance",
    "accountant": "finance",
    "finance manager": "finance",
    "investment banker": "finance",
    "auditor": "finance",
    "tax consultant": "finance",
    "equity analyst": "finance",
    "credit analyst": "finance",
    "loan officer": "finance",

    # ===================== MANAGEMENT =====================

    "business analyst": "management",
    "project manager": "management",
    "operations manager": "management",
    "product manager": "management",
    "program manager": "management",
    "team lead": "management",
    "scrum master": "management",

    # ===================== MARKETING =====================

    "marketing manager": "marketing",
    "digital marketer": "marketing",
    "seo specialist": "marketing",
    "content writer": "marketing",
    "social media manager": "marketing",
    "brand manager": "marketing",
    "performance marketer": "marketing",

    # ===================== HR =====================

    "hr": "hr",
    "hr manager": "hr",
    "hr executive": "hr",
    "recruiter": "hr",
    "talent acquisition specialist": "hr",
    "hr recruiter": "hr",
    "technical recruiter": "hr",

    # ===================== SALES =====================

    "sales executive": "sales",
    "sales manager": "sales",
    "business development executive": "sales",
    "business development manager": "sales",
    "bdm": "sales",
    "inside sales executive": "sales",
    "account manager": "sales",

    # ===================== OPERATIONS =====================

    "operations executive": "operations",
    "operations manager": "operations",
    "supply chain manager": "operations",
    "logistics manager": "operations",
    "procurement specialist": "operations",

    # ===================== DESIGN =====================

    "ui ux designer": "design",
    "ui designer": "design",
    "ux designer": "design",
    "graphic designer": "design",
    "product designer": "design",
    "visual designer": "design",

    # ===================== CYBERSECURITY =====================

    "cyber security engineer": "cybersecurity",
    "cybersecurity analyst": "cybersecurity",
    "ethical hacker": "cybersecurity",
    "penetration tester": "cybersecurity",
    "security analyst": "cybersecurity",
    "information security analyst": "cybersecurity",

    # ===================== HEALTHCARE =====================

    "doctor": "healthcare",
    "physician": "healthcare",
    "nurse": "healthcare",
    "staff nurse": "healthcare",
    "medical officer": "healthcare",
    "surgeon": "healthcare",
    "dentist": "healthcare",
    "pharmacist": "healthcare",
    "clinical researcher": "healthcare",
    "medical coder": "healthcare",
    "healthcare assistant": "healthcare",
    "hospital administrator": "healthcare",
    "lab technician": "healthcare",
    "pathologist": "healthcare",
    "radiologist": "healthcare",
    "medical imaging specialist": "healthcare",
    "epidemiologist": "healthcare",
    "biotech analyst": "healthcare",
    "healthcare manager": "healthcare",
    "clinical data analyst": "healthcare",
    "therapist": "healthcare",
    "physiotherapist": "healthcare",
    "nutritionist": "healthcare",
    "public health specialist": "healthcare",
    "medical representative": "healthcare",
    "clinical pharmacist": "healthcare",
    "healthcare consultant": "healthcare",

    # ===================== ENGINEERING =====================

    "mechanical engineer": "engineering",
    "civil engineer": "engineering",
    "electrical engineer": "engineering",
    "electronics engineer": "engineering",
    "design engineer": "engineering",
    "site engineer": "engineering",
    "construction engineer": "engineering",

    # ===================== LEGAL =====================

    "lawyer": "legal",
    "legal advisor": "legal",
    "legal consultant": "legal",
    "corporate lawyer": "legal",
    "advocate": "legal",
    "legal associate": "legal",

    # ===================== EDUCATION =====================

    "teacher": "education",
    "professor": "education",
    "lecturer": "education",
    "trainer": "education",
    "educator": "education",
    "academic coordinator": "education"
}
if st.button('Analyze'):
    if pdf is not None:
        if jd.strip()!="":
            # reset pointer (important)
            pdf.seek(0)
            resume_text = extract_text_from_pdf(pdf)
            resume_text = clean_text(resume_text)
            resume_embedding = embed_model.encode([resume_text])
            prediction = model.predict(resume_embedding)
            role = encode.inverse_transform(prediction)
            jd=clean_text(jd)
            jd_embed=embed_model.encode([jd])
            similarity=cosine_similarity(resume_embedding,jd_embed)
            ats_score=round(similarity[0][0]*100,2)
            if ats_score >= 80:
                st.success(f"Excellent ATS Match: {ats_score}%")
            elif ats_score >= 50:
                st.info(f"Good ATS Match: {ats_score}%")
            elif ats_score >= 35:
                st.warning(f"Average ATS Match: {ats_score}%")
            else:
                st.error(f"Weak ATS Match")
            st.success(f"My AI model says You are Suitable for {role[0].upper()} profile")
        else:
            st.error("Enter the job description")
    else:
        st.error("Please upload resume")
st.sidebar.title('Home')
if(st.sidebar.button('Skill Gap Analysis')):
    if pdf is not None and jd.strip() != "":
        resume_text = extract_text_from_pdf(pdf)
        resume_text = clean_text(resume_text)
        resume_embedding = embed_model.encode([resume_text])
        prediction = model.predict(resume_embedding)
        role = encode.inverse_transform(prediction)
        pdf.seek(0) #reset cursor pointer 
        resume_text = extract_text_from_pdf(pdf)
        resume_text = clean_text(resume_text)
        jd = clean_text(jd)
        detected_category = None
        for role_name, category in role_mapping.items():
            if role_name in jd:
                detected_category = category
                break
        if detected_category is None:
            category_score = {}
            for category, subcategories in skills_db.items():
                count = 0
                for subcategory, skills in subcategories.items():
                    for skill in skills:
                        if skill.lower() in jd:
                            count += 1
                category_score[category] = count
            # no match found
            if max(category_score.values()) == 0:
                detected_category = "Unknown"
            else:
                detected_category = max(category_score,key=category_score.get)
        st.success(f"Your Domain is: {detected_category.upper()}")
        if detected_category != "Unknown":
            required_skills = skills_db[detected_category]
            matched_skills = {}
            missing_skills = {}
            for subcategory, skills in required_skills.items():
                matched_skills[subcategory] = []
                missing_skills[subcategory] = []
                for skill in skills:
                    if skill.lower() in resume_text:
                        matched_skills[subcategory].append(skill)
                    else:
                        missing_skills[subcategory].append(skill)
            st.markdown("""<h1 style='color:red'>Matched Skills</h1>""",unsafe_allow_html=True)
            for subcategory, skills in matched_skills.items():
                if len(skills) > 0:
                    st.write(f"### {subcategory}")
                    st.write(skills)
            st.markdown("""<h1 style='color:red'>Missing Skills</h1>""",unsafe_allow_html=True)
            for subcategory, skills in missing_skills.items():
                if len(skills) > 0:
                    st.write(f"### {subcategory}")
                    st.write(skills)
            matched_count = sum(
                len(v) for v in matched_skills.values()
            )
            missing_count = sum(
                len(v) for v in missing_skills.values()
            )
            def generate_pdf_report(role,category, matched_skills, missing_skills, matched_count, missing_count):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                content = []
                content.append(Paragraph("AI Resume Skill Analysis Report", styles['Title']))
                content.append(Spacer(1, 12))
                content.append(Paragraph(f"Best Matching Career Domain: <b>{category}</b>", styles['Normal']))
                content.append(Spacer(1, 12))
                content.append(Paragraph("Matched Skills:", styles['Heading2']))
                for sub, skills in matched_skills.items():
                    if skills:
                        content.append(Paragraph(f"<b>{sub}</b>: {', '.join(skills)}", styles['Normal']))
                        content.append(Spacer(1, 10))
                    # Missing Skills
                content.append(Paragraph("Missing Skills:", styles['Heading2']))
                for sub, skills in missing_skills.items():
                    if skills:
                        content.append(Paragraph(f"<b>{sub}</b>: {', '.join(skills)}", styles['Normal']))
                        content.append(Spacer(1, 12))
                content.append(Paragraph(f"Total Matched Skills: {matched_count}", styles['Normal']))
                content.append(Paragraph(f"Total Missing Skills: {missing_count}", styles['Normal']))
                content.append(Paragraph(f"My AI model says You are suitable for:{role[0].upper()} profile ",styles['Heading2']))
                doc.build(content)
                buffer.seek(0)
                return buffer
            #Graph visualize
            labels = ['Matched', 'Missing']
            values = [matched_count, missing_count]
            fig, ax = plt.subplots(figsize=(6,5))
            ax.bar(labels, values)
            plt.title("Resume Skill Analysis")
            plt.xlabel("Skill Type")
            plt.ylabel("Number of Skills")
            st.sidebar.pyplot(fig)
            st.sidebar.snow()
            pdf_report = generate_pdf_report(role,detected_category,matched_skills,missing_skills,matched_count,missing_count)

            st.sidebar.download_button(label="📄 Download Report as PDF",data=pdf_report,file_name="resume_skill_report.pdf",mime="application/pdf")
        else:
            st.warning("No matching category found")
    else:
        st.error("Please Fill Both Fields")
