import streamlit as st
import pandas as pd

st.set_page_config(page_title="SkillBridge | Agentic RAG", page_icon="🎓", layout="wide")

st.title("🎓 SkillBridge — Student Academic Intelligence")
st.caption("Agentic RAG Challenge · Student performance dashboard")
st.info(
    "Upload the challenge CSV files to explore student, course, assessment and attendance data. "
    "The Kaggle notebook contains the Agentic RAG pipeline; this app currently provides the dashboard layer."
)

with st.sidebar:
    st.header("Upload data")
    students_file = st.file_uploader("students.csv", type="csv")
    courses_file = st.file_uploader("courses.csv", type="csv")
    attendance_file = st.file_uploader("attendance.csv", type="csv")
    assessments_file = st.file_uploader("assessments.csv", type="csv")
    mentors_file = st.file_uploader("mentors.csv", type="csv")

def read_csv(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None

students = read_csv(students_file)
courses = read_csv(courses_file)
attendance = read_csv(attendance_file)
assessments = read_csv(assessments_file)
mentors = read_csv(mentors_file)

frames = {
    "Students": students, "Courses": courses, "Attendance": attendance,
    "Assessments": assessments, "Mentors": mentors
}
loaded = {name: df for name, df in frames.items() if df is not None}

if not loaded:
    st.subheader("Welcome")
    st.write("Start by uploading one or more CSV files from the challenge's `Dataset` folder.")
    st.markdown("""
    **Expected files**
    - `students.csv`
    - `courses.csv`
    - `attendance.csv`
    - `assessments.csv`
    - `mentors.csv`
    """)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Datasets loaded", len(loaded))
c2.metric("Students", len(students) if students is not None else "—")
c3.metric("Courses", len(courses) if courses is not None else "—")

st.divider()
st.subheader("Data overview")
tabs = st.tabs(list(loaded.keys()))
for tab, (name, df) in zip(tabs, loaded.items()):
    with tab:
        st.write(f"**{len(df):,} rows · {len(df.columns)} columns**")
        st.dataframe(df.head(100), use_container_width=True)

st.divider()
st.subheader("Find a student")
if students is None:
    st.warning("Upload students.csv to enable student search.")
else:
    possible_name_cols = ["student_name", "name"]
    possible_id_cols = ["student_id", "id"]
    name_col = next((c for c in possible_name_cols if c in students.columns), None)
    id_col = next((c for c in possible_id_cols if c in students.columns), None)
    if name_col or id_col:
        search = st.text_input("Search by student name or ID")
        if search:
            mask = pd.Series(False, index=students.index)
            for col in [name_col, id_col]:
                if col:
                    mask |= students[col].astype(str).str.contains(search, case=False, na=False)
            matches = students.loc[mask]
            if matches.empty:
                st.warning("No matching student found.")
            else:
                st.dataframe(matches, use_container_width=True)
                if attendance is not None and id_col and "student_id" in attendance.columns:
                    chosen = st.selectbox(
                        "Select student to inspect attendance",
                        matches[id_col].astype(str).tolist()
                    )
                    rows = attendance[attendance["student_id"].astype(str) == chosen]
                    if not rows.empty and "status" in rows.columns:
                        status = rows["status"].astype(str).str.lower()
                        total = len(rows)
                        present = int((status == "present").sum())
                        late = int((status == "late").sum())
                        absent = int((status == "absent").sum())
                        a, b, c, d = st.columns(4)
                        a.metric("Sessions", total)
                        b.metric("Present", present)
                        c.metric("Late", late)
                        d.metric("Absent", absent)
                        st.caption("Raw attendance counts only; weighted attendance must follow the supplied policy.")
                        st.dataframe(rows, use_container_width=True)
    else:
        st.error("Could not identify student name/ID columns. Check the uploaded CSV headers.")

st.divider()
st.subheader("Ask Edu/SkillBridge")
st.text_area("Your question", placeholder="Example: What is the attendance status of student STU0545?")
st.caption(
    "The interactive Agentic RAG question-answering endpoint is not connected in this dashboard yet. "
    "Run the notebook in Kaggle for the full retrieval + tool-calling workflow and submission generation."
)
