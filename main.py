
import os
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import fitz

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv(override=True)

st.set_page_config(
    page_title="AI CareerMate",
    page_icon="💼",
    layout="wide"
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    st.error("Gemini API key is missing. Check your .env file.")
    st.stop()

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=45000,
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1,
            max_delay=3,
            exp_base=2,
            jitter=0.2
        )
    )
)


# ==========================================
# 2. SESSION STATE
# ==========================================

defaults = {
    "questions": [],
    "answers": [],
    "evaluations": [],
    "current_question": 0,
    "interview_started": False,
    "interview_finished": False,
    "interview_saved": False,
    "saved_interview_id": None,
    "job_role": "",
    "experience": "",
    "difficulty": "",
    "resume_result": "",
    "roadmap_result": ""
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 2A. PERSISTENT INTERVIEW HISTORY (SQLITE)
# ==========================================

DB_PATH = Path(__file__).resolve().parent / "careermate_history.db"


def get_db_connection():
    return sqlite3.connect(DB_PATH, timeout=15)


def initialize_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                experience TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                evaluations_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                average_score REAL NOT NULL,
                final_report TEXT DEFAULT ''
            )
            """
        )
        connection.commit()


def extract_score(evaluation):
    """Read a score such as 'Score: 8/10' from the AI feedback."""
    patterns = [
        r"score\s*[:\-]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10"
    ]
    for pattern in patterns:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            return max(0.0, min(10.0, float(match.group(1))))
    return None


def save_completed_interview():
    """Save a completed interview only once, despite Streamlit reruns."""
    if st.session_state.get("interview_saved", False):
        return st.session_state.get("saved_interview_id")

    questions = st.session_state.questions
    answers = st.session_state.answers
    evaluations = st.session_state.evaluations
    scores = [extract_score(item) for item in evaluations]
    valid_scores = [score for score in scores if score is not None]
    average = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO interview_history (
                created_at, role, experience, difficulty,
                questions_json, answers_json, evaluations_json,
                scores_json, average_score, final_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                st.session_state.job_role,
                st.session_state.experience,
                st.session_state.difficulty,
                json.dumps(questions, ensure_ascii=False),
                json.dumps(answers, ensure_ascii=False),
                json.dumps(evaluations, ensure_ascii=False),
                json.dumps(scores),
                average,
                ""
            )
        )
        connection.commit()
        saved_id = cursor.lastrowid

    st.session_state.interview_saved = True
    st.session_state.saved_interview_id = saved_id
    return saved_id


def update_saved_report(interview_id, report):
    if not interview_id:
        return
    with get_db_connection() as connection:
        connection.execute(
            "UPDATE interview_history SET final_report = ? WHERE id = ?",
            (report, interview_id)
        )
        connection.commit()


def load_interview_history():
    with get_db_connection() as connection:
        frame = pd.read_sql_query(
            """
            SELECT id, created_at, role, experience, difficulty,
                   questions_json, answers_json, evaluations_json,
                   scores_json, average_score, final_report
            FROM interview_history
            ORDER BY id DESC
            """,
            connection
        )
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
        frame["scores"] = frame["scores_json"].apply(
            lambda value: json.loads(value) if value else []
        )
    return frame


initialize_database()


# Small dashboard styling polish.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 3. GEMINI AI FUNCTION
# =========================================

import time

def ask_ai(prompt):

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError("AI returned an empty response.")

    return response.text

    


# ==========================================
# 4. GENERATE INTERVIEW QUESTIONS
# ==========================================

def generate_questions(role, experience, difficulty):

    prompt = f"""
    You are an expert professional interviewer.

    Create exactly 5 interview questions for:

    Job role: {role}
    Experience: {experience}
    Difficulty: {difficulty}

    Requirements:
    - Questions must be relevant to the selected role.
    - Include practical and conceptual questions.
    - Avoid duplicate questions.
    - Match the selected difficulty.

    Return ONLY valid JSON in this format:

    {{
        "questions": [
            "Question 1",
            "Question 2",
            "Question 3",
            "Question 4",
            "Question 5"
        ]
    }}
    """

    result = ask_ai(prompt)

    data = json.loads(result)

    questions = data["questions"]

    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("AI did not generate exactly five questions.")

    return questions


# ==========================================
# 5. EVALUATE ANSWER
# ==========================================

def evaluate_answer(question, answer, role):

    prompt = f"""
    You are a professional interview evaluator.

    Job role: {role}

    Interview question:
    {question}

    Candidate's answer:
    {answer}

    Evaluate the answer honestly.

    Consider:
    1. Technical accuracy.
    2. Relevance.
    3. Clarity.
    4. Completeness.

    Give a score out of 10.

    Use this format:

    ### Score
    Score: X/10

    ### Strengths
    Explain what the candidate did well.

    ### Weaknesses
    Explain what is missing or incorrect.

    ### Improved Answer
    Write a better example answer in simple language.

    ### Interview Tip
    Give one practical improvement tip.
    """

    return ask_ai(prompt)


# ==========================================
# 6. FINAL INTERVIEW REPORT
# ==========================================

def generate_report(questions, answers, evaluations):

    report_data = ""

    for i in range(len(questions)):

        report_data += f"""
        Question: {questions[i]}

        Candidate Answer: {answers[i]}

        Evaluation: {evaluations[i]}
        """

    prompt = f"""
    You are a professional interview coach.

    Review this complete interview.

    {report_data}

    Prepare a final performance report.

    Include:
    1. Overall performance summary.
    2. Strongest areas.
    3. Weakest areas.
    4. Technical knowledge assessment.
    5. Communication assessment.
    6. Three improvement recommendations.

    Give an overall score out of 100.
    Be fair and constructive.
    """

    return ask_ai(prompt)


# ==========================================
# 7. EXTRACT PDF TEXT
# ==========================================

def extract_resume_text(uploaded_file):

    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:

        text += page.get_text("text") + "\n"

    pdf.close()

    return text


# ==========================================
# 8. RESUME ANALYZER
# ==========================================

def analyze_resume(resume_text, target_role):

    prompt = f"""
    You are an experienced career advisor and resume reviewer.

    Target job role: {target_role}

    Analyze the following resume.

    RESUME:
    {resume_text[:25000]}

    Give feedback under these headings:

    1. Resume Overview
    2. Skills Identified
    3. Strengths
    4. Missing or Weak Skills
    5. Formatting and Presentation
    6. Suggestions for Improvement
    7. Recommended Next Steps

    Be constructive and specific.

    Do not invent qualifications or experience.
    Do not claim that an applicant will definitely get a job.
    """

    return ask_ai(prompt)


# ==========================================
# 9. CAREER ROADMAP GENERATOR
# ==========================================

def generate_roadmap(career, level):

    prompt = f"""
    You are a professional career mentor.

    Create a personalized career roadmap.

    Desired career: {career}
    Current level: {level}

    Include:

    1. Career introduction.
    2. Essential technical skills.
    3. Programming languages and tools.
    4. Learning stages in the correct order.
    5. Five practical projects.
    6. Portfolio and GitHub recommendations.
    7. Interview preparation.
    8. Suggested learning timeline.

    Use simple language.
    Make the roadmap realistic for the selected level.
    Keep the complete roadmap concise and practical.
    Use bullet points.
    Avoid lengthy explanations.
    Maximum 700 words.
    """

    return ask_ai(prompt)


# ==========================================
# 10. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("💼 CareerMate")

st.sidebar.caption("Your AI Interview & Career Coach")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Mock Interview",
        "Interview History",
        "Resume Analyzer",
        "Career Roadmap"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Powered by Gemini AI")


# ==========================================
# 11. DASHBOARD WITH INTERVIEW ANALYTICS
# ==========================================

if page == "Dashboard":

    history = load_interview_history()

    st.title("💼 Welcome to AI CareerMate")
    st.write("Your personal AI interview and career companion.")

    if history.empty:
        total_interviews = 0
        average_score = 0.0
        best_score = 0.0
        latest_role = "Not yet available"
    else:
        total_interviews = len(history)
        average_score = float(history["average_score"].mean())
        best_score = float(history["average_score"].max())
        latest_role = str(history.iloc[0]["role"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interviews completed", total_interviews)
    col2.metric("Average score", f"{average_score:.1f}/10")
    col3.metric("Best interview score", f"{best_score:.1f}/10")
    col4.metric("Latest job role", latest_role)

    st.divider()
    st.subheader("Explore CareerMate")

    left, right = st.columns(2)
    with left:
        st.info("### 🎯 Mock Interview\nPractice role-specific questions and receive AI feedback.")
        st.info("### 📄 Resume Analyzer\nUpload your resume and receive professional feedback.")
    with right:
        st.info("### 📊 Interview Analytics\nReview your scores and track your progress.")
        st.info("### 🚀 Career Roadmap\nDiscover the skills needed for your dream career.")

    st.divider()
    st.subheader("📈 Your interview performance")

    if history.empty:
        st.info("Your analytics will appear here after your first completed mock interview.")
    else:
        chart_data = history.sort_values("created_at").copy()
        chart_data["Interview number"] = range(1, len(chart_data) + 1)

        chart = px.line(
            chart_data,
            x="Interview number",
            y="average_score",
            markers=True,
            hover_data=["role", "created_at"],
            title="Average score across your interviews",
            labels={"average_score": "Average score out of 10"}
        )
        chart.update_yaxes(range=[0, 10])
        chart.update_layout(height=350)
        st.plotly_chart(chart, use_container_width=True)

        recent = history.head(5).copy()
        recent["Date"] = recent["created_at"].dt.strftime("%d %b %Y, %I:%M %p")
        recent = recent.rename(
            columns={
                "role": "Job role",
                "difficulty": "Difficulty",
                "average_score": "Average score / 10"
            }
        )
        st.subheader("Recent interviews")
        st.dataframe(
            recent[["Date", "Job role", "Difficulty", "Average score / 10"]],
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.caption("AI CareerMate | Built with Python, Streamlit, SQLite, Plotly and Gemini AI")


# 12. MOCK INTERVIEW
# ==========================================

elif page == "Mock Interview":

    st.title("🎯 AI Mock Interview")

    st.write(
        "Practice real-world interview questions "
        "and improve your answers."
    )

    # INTERVIEW SETTINGS

    if not st.session_state.interview_started:

        role = st.selectbox(
            "Select your job role",
            [
                "Python Developer",
                "Frontend Developer",
                "Backend Developer",
                "Data Analyst",
                "AI / ML Engineer",
                "Software Engineer",
                "Cybersecurity Analyst",
                "Database Administrator",
                "Digital Marketer",
                "Business Analyst",
                "HR Executive",
                "Other"
            ]
        )

        if role == "Other":

            role = st.text_input(
                "Enter your desired job role"
            )

        experience = st.selectbox(
            "Experience level",
            [
                "Student / Fresher",
                "Beginner",
                "Intermediate",
                "Experienced"
            ]
        )

        difficulty = st.selectbox(
            "Interview difficulty",
            [
                "Easy",
                "Medium",
                "Hard"
            ]
        )

        if st.button(
            "🚀 Start Interview",
            type="primary"
        ):

            if not role.strip():

                st.warning("Please enter a job role.")

            else:

                with st.spinner(
                    "CareerMate is preparing your interview..."
                ):

                    try:

                        questions = generate_questions(
                            role,
                            experience,
                            difficulty
                        )

                        st.session_state.questions = questions
                        st.session_state.answers = []
                        st.session_state.evaluations = []

                        st.session_state.current_question = 0

                        st.session_state.job_role = role
                        st.session_state.experience = experience
                        st.session_state.difficulty = difficulty

                        st.session_state.interview_started = True
                        st.session_state.interview_saved = False
                        st.session_state.saved_interview_id = None
                        st.session_state.final_report = ""
                        st.session_state.interview_finished = False

                        st.rerun()

                    except Exception as e:

                        st.error(f"Could not start interview: {e}")


    # INTERVIEW QUESTIONS

    elif (
        st.session_state.interview_started
        and not st.session_state.interview_finished
    ):

        total = len(st.session_state.questions)

        current = st.session_state.current_question

        st.progress(
            current / total
        )

        st.caption(
            f"Question {current + 1} of {total}"
        )

        st.subheader(
            st.session_state.questions[current]
        )

        with st.form("answer_form"):

            answer = st.text_area(
                "Your Answer",
                placeholder="Write your interview answer here...",
                height=180
            )

            submitted = st.form_submit_button(
                "Submit Answer",
                type="primary"
            )

        if submitted:

            if not answer.strip():

                st.warning("Please enter your answer.")

            else:

                with st.spinner(
                    "AI is evaluating your answer..."
                ):

                    try:

                        evaluation = evaluate_answer(
                            st.session_state.questions[current],
                            answer,
                            st.session_state.job_role
                        )

                        st.session_state.answers.append(answer)

                        st.session_state.evaluations.append(
                            evaluation
                        )

                        st.session_state.current_question += 1

                        if (
                            st.session_state.current_question
                            >= total
                        ):

                            st.session_state.interview_finished = True
                            try:
                                save_completed_interview()
                            except Exception as db_error:
                                st.error(f"Interview completed, but history could not be saved: {db_error}")

                        st.rerun()

                    except Exception as e:

                        st.error(f"Evaluation error: {e}")

        if st.session_state.evaluations:

            st.divider()

            st.subheader("Previous Feedback")

            for i, evaluation in enumerate(
                st.session_state.evaluations
            ):

                with st.expander(
                    f"Feedback for Question {i + 1}"
                ):

                    st.markdown(evaluation)


    # FINAL INTERVIEW REPORT

    elif st.session_state.interview_finished:

        st.title("🏆 Interview Completed!")

        st.success(
            "Congratulations! You have completed your mock interview."
        )


        scores = [extract_score(item) for item in st.session_state.evaluations]
        valid_scores = [score for score in scores if score is not None]

        if valid_scores:
            average = sum(valid_scores) / len(valid_scores)
            st.metric("Your average answer score", f"{average:.1f}/10")

            score_frame = pd.DataFrame({
                "Question": [f"Q{i + 1}" for i in range(len(scores))],
                "Score": [score if score is not None else 0 for score in scores]
            })
            score_chart = px.bar(
                score_frame,
                x="Question",
                y="Score",
                text="Score",
                title="Your question-wise performance"
            )
            score_chart.update_yaxes(range=[0, 10])
            st.plotly_chart(score_chart, use_container_width=True)

        if not st.session_state.get("interview_saved", False):
            if st.button("Save interview to history"):
                try:
                    save_completed_interview()
                    st.success("Interview saved to your history.")
                    st.rerun()
                except Exception as db_error:
                    st.error(f"Could not save interview: {db_error}")

        for i, evaluation in enumerate(
            st.session_state.evaluations
        ):

            with st.expander(
                f"Question {i + 1} – Evaluation",
                expanded=False
            ):

                st.markdown(evaluation)

        st.divider()

        st.subheader("📊 Your Final Performance Report")

        if st.button(
            "Generate Final Report",
            type="primary"
        ):

            with st.spinner(
                "Preparing your performance report..."
            ):

                try:

                    report = generate_report(
                        st.session_state.questions,
                        st.session_state.answers,
                        st.session_state.evaluations
                    )

                    st.session_state.final_report = report
                    update_saved_report(
                        st.session_state.get("saved_interview_id"), report
                    )

                except Exception as e:

                    st.error(f"Report error: {e}")

        if "final_report" in st.session_state:

            st.markdown(
                st.session_state.final_report
            )

            st.download_button(
                "Download Interview Report",
                data=st.session_state.final_report,
                file_name="CareerMate_Interview_Report.txt",
                mime="text/plain"
            )

        if st.button("Start a New Interview"):

            st.session_state.interview_started = False
            st.session_state.interview_finished = False
            st.session_state.interview_saved = False
            st.session_state.saved_interview_id = None

            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.evaluations = []

            st.session_state.current_question = 0

            st.session_state.pop("final_report", None)

            st.rerun()


# ==========================================
# 13. INTERVIEW HISTORY
# ==========================================

elif page == "Interview History":

    st.title("🗂️ Interview History")
    st.write("Review your previous mock interviews and their performance.")

    history = load_interview_history()

    if history.empty:
        st.info("No saved interviews yet. Complete a mock interview to build your history.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total saved interviews", len(history))
        col2.metric("Overall average", f"{history['average_score'].mean():.1f}/10")

        st.divider()

        for _, record in history.iterrows():
            date_value = record["created_at"]
            date_label = (
                date_value.strftime("%d %b %Y, %I:%M %p")
                if pd.notna(date_value) else "Date unavailable"
            )

            with st.expander(
                f"{record['role']} · {date_label} · {record['average_score']:.1f}/10"
            ):
                st.write(f"**Experience:** {record['experience']}")
                st.write(f"**Difficulty:** {record['difficulty']}")
                st.write(f"**Average score:** {record['average_score']:.1f}/10")

                questions = json.loads(record["questions_json"])
                answers = json.loads(record["answers_json"])
                evaluations = json.loads(record["evaluations_json"])
                scores = record["scores"]

                score_rows = []
                for index, question in enumerate(questions):
                    score = scores[index] if index < len(scores) else None
                    score_rows.append({
                        "Question": f"Q{index + 1}",
                        "Interview question": question,
                        "Score / 10": score if score is not None else "Not detected"
                    })

                st.dataframe(
                    pd.DataFrame(score_rows),
                    use_container_width=True,
                    hide_index=True
                )

                for index, evaluation in enumerate(evaluations):
                    with st.expander(f"Question {index + 1} feedback"):
                        st.markdown(f"**Your answer:** {answers[index]}")
                        st.markdown(evaluation)

                if record["final_report"]:
                    st.markdown("### Final report")
                    st.markdown(record["final_report"])
                    st.download_button(
                        "Download this interview report",
                        data=record["final_report"],
                        file_name=f"CareerMate_Interview_{record['id']}.txt",
                        mime="text/plain",
                        key=f"history_report_{record['id']}"
                    )


# 14. RESUME ANALYZER
# ==========================================

elif page == "Resume Analyzer":

    st.title("📄 AI Resume Analyzer")

    st.write(
        "Upload your resume and get AI-powered career feedback."
    )

    target_role = st.selectbox(
        "What job are you applying for?",
        [
            "Python Developer",
            "Software Engineer",
            "Data Analyst",
            "AI / ML Engineer",
            "Frontend Developer",
            "Cybersecurity Analyst",
            "Business Analyst",
            "Other"
        ]
    )

    if target_role == "Other":

        target_role = st.text_input(
            "Enter your target job"
        )

    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "Analyze My Resume",
            type="primary"
        ):

            with st.spinner(
                "CareerMate is analyzing your resume..."
            ):

                try:

                    resume_text = extract_resume_text(
                        uploaded_file
                    )

                    if not resume_text.strip():

                        st.warning(
                            "No text found. "
                            "This may be a scanned PDF."
                        )

                    else:

                        result = analyze_resume(
                            resume_text,
                            target_role
                        )

                        st.session_state.resume_result = result

                except Exception as e:

                    st.error(f"Resume error: {e}")

    if st.session_state.resume_result:

        st.divider()

        st.subheader("Your Resume Analysis")

        st.markdown(
            st.session_state.resume_result
        )

        st.download_button(
            "Download Resume Feedback",
            data=st.session_state.resume_result,
            file_name="Resume_Feedback.txt",
            mime="text/plain"
        )


# ==========================================
# 15. CAREER ROADMAP
# ==========================================

elif page == "Career Roadmap":

    st.title("🚀 AI Career Roadmap Generator")

    st.write(
        "Discover what you need to learn to reach your dream career."
    )

    career = st.selectbox(
        "Choose your desired career",
        [
            "Python Developer",
            "Full Stack Developer",
            "Data Scientist",
            "AI / ML Engineer",
            "Cybersecurity Analyst",
            "Data Analyst",
            "Mobile App Developer",
            "UI / UX Designer",
            "Cloud Engineer",
            "Software Engineer",
            "Other"
        ]
    )

    if career == "Other":

        career = st.text_input(
            "Enter your desired career"
        )

    level = st.selectbox(
        "Your current skill level",
        [
            "Complete Beginner",
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    if st.button(
        "Generate My Roadmap",
        type="primary"
    ):

        if not career.strip():

            st.warning("Please enter a career.")

        else:

            with st.spinner(
                "Creating your personalized roadmap..."
            ):

                try:

                    result = generate_roadmap(
                        career,
                        level
                    )

                    st.session_state.roadmap_result = result

                except Exception as e:

                    st.error(f"Roadmap error: {e}")

    if st.session_state.roadmap_result:

        st.divider()

        st.subheader("Your Personalized Career Roadmap")

        st.markdown(
            st.session_state.roadmap_result
        )

        st.download_button(
            "Download Career Roadmap",
            data=st.session_state.roadmap_result,
            file_name="Career_Roadmap.txt",
            mime="text/plain"
        )


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI CareerMate | Built with Python, Streamlit and Gemini AI"
)
