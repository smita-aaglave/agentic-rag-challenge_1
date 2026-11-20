
import os
import json
import re
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import streamlit as st

from google import genai
from google.genai import types
from openai import OpenAI


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POLICY_DIR = DATA_DIR / "Policy Documents"

LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "gemini-embedding-001"

REFERENCE_DATE = pd.Timestamp("2025-01-15")


# =====================================================
# API KEYS
# =====================================================

def get_api_keys():

    groq_keys = []

    try:
        for i in range(1, 9):
            key = st.secrets.get(f"GROQ_API_KEY_{i}", "")

            if key:
                key = str(key).strip()

                if key and key not in groq_keys:
                    groq_keys.append(key)

        google_key = str(
            st.secrets.get("GOOGLE_API_KEY", "")
        ).strip()

    except Exception as e:
        raise RuntimeError(
            "Streamlit secrets not found. "
            "Configure GOOGLE_API_KEY and GROQ_API_KEY_1 "
            "in .streamlit/secrets.toml."
        ) from e

    if not groq_keys:
        raise RuntimeError("Groq API key is missing.")

    if not google_key:
        raise RuntimeError("Google API key is missing.")

    return groq_keys, google_key


# =====================================================
# API KEY POOL
# =====================================================

class KeyPool:

    def __init__(self, keys, rpm_per_key=20):

        self.keys = keys
        self.rpm = rpm_per_key

        self.calls = {key: [] for key in keys}
        self.cooldown_until = {key: 0 for key in keys}

        self.idx = 0

    def get(self):

        while True:

            now = time.time()

            for _ in range(len(self.keys)):

                key = self.keys[self.idx]

                self.idx = (self.idx + 1) % len(self.keys)

                self.calls[key] = [
                    t for t in self.calls[key]
                    if now - t < 60
                ]

                if (
                    now >= self.cooldown_until[key]
                    and len(self.calls[key]) < self.rpm
                ):

                    self.calls[key].append(now)

                    return key

            time.sleep(1)

    def penalize(self, key, seconds=30):

        self.cooldown_until[key] = time.time() + seconds


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    required_files = [
        "courses.csv",
        "mentors.csv",
        "students.csv",
        "assessments.csv",
        "attendance.csv"
    ]

    for filename in required_files:

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset missing: {path}"
            )

    courses_df = pd.read_csv(DATA_DIR / "courses.csv")
    mentors_df = pd.read_csv(DATA_DIR / "mentors.csv")
    students_df = pd.read_csv(DATA_DIR / "students.csv")
    assessments_df = pd.read_csv(DATA_DIR / "assessments.csv")
    attendance_df = pd.read_csv(DATA_DIR / "attendance.csv")

    policy_docs = {}

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy folder not found: {POLICY_DIR}"
        )

    for file in POLICY_DIR.glob("*.md"):

        policy_docs[file.name] = file.read_text(
            encoding="utf-8"
        )

    if not policy_docs:
        raise RuntimeError("No policy documents found.")

    return (
        courses_df,
        mentors_df,
        students_df,
        assessments_df,
        attendance_df,
        policy_docs
    )


# =====================================================
# GEMINI + FAISS RAG
# =====================================================

def create_rag(gemini_client, policy_docs):

    policy_chunks = []

    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200

    for filename, content in policy_docs.items():

        content = content.strip()
        start = 0

        while start < len(content):

            end = start + CHUNK_SIZE

            chunk_text = content[start:end].strip()

            if chunk_text:

                policy_chunks.append({
                    "source": filename,
                    "text": chunk_text
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    if not policy_chunks:
        raise RuntimeError("No policy chunks created.")

    all_embeddings = []

    for i in range(0, len(policy_chunks), 16):

        batch = policy_chunks[i:i + 16]

        response = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=[chunk["text"] for chunk in batch],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        all_embeddings.extend([
            item.values for item in response.embeddings
        ])

    embedding_matrix = np.array(
        all_embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(embedding_matrix)

    index = faiss.IndexFlatIP(
        embedding_matrix.shape[1]
    )

    index.add(embedding_matrix)

    return policy_chunks, index


def retrieve_policies(
    query,
    gemini_client,
    policy_chunks,
    faiss_index,
    top_k=3
):

    response = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    query_embedding = np.array(
        [response.embeddings[0].values],
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(
        query_embedding,
        min(top_k, len(policy_chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx >= 0:

            results.append({
                "source": policy_chunks[idx]["source"],
                "text": policy_chunks[idx]["text"],
                "score": float(score)
            })

    return results


# =====================================================
# GROQ LLM
# =====================================================

def call_llm(messages, tools, pool):

    for attempt in range(6):

        key = pool.get()

        client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1"
        )

        request_args = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.0
        }

        # Tool settings are included only when tools exist.
        if tools:

            request_args["tools"] = tools
            request_args["tool_choice"] = "auto"

        try:

            return client.chat.completions.create(
                **request_args
            )

        except Exception as e:

            error_message = str(e).lower()

            if (
                "429" in error_message
                or "rate" in error_message
                or "limit" in error_message
            ):

                pool.penalize(key, 30)

            if attempt == 5:

                raise RuntimeError(
                    f"Groq API failed: {e}"
                ) from e

            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError("LLM failed after retries.")


# =====================================================
# ATTENDANCE CALCULATIONS
# =====================================================

def calculate_weighted_attendance(attendance_df):

    if attendance_df.empty:
        return 0.0, 0.0

    session_types = (
        attendance_df["session_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    statuses = (
        attendance_df["status"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    weights = session_types.map(
        lambda value: 1.5 if value == "lab" else 1.0
    )

    credits = statuses.map(
        lambda value:
        1.0 if value == "present"
        else 0.5 if value == "late"
        else 0.0
    )

    weighted_total = float(weights.sum())

    weighted_present = float(
        (weights * credits).sum()
    )

    percentage = (
        weighted_present / weighted_total * 100
        if weighted_total > 0
        else 0.0
    )

    return percentage, weighted_total


# =====================================================
# AGENT TOOLS
# =====================================================

def create_agent_tools(
    courses_df,
    students_df,
    assessments_df,
    attendance_df
):

    # ---------------------------------------------
    # FIND STUDENT
    # ---------------------------------------------

    def find_student(student_name=None, student_id=None):

        if student_id:
            matches = students_df[
                students_df["student_id"].astype(str).str.strip()
                == str(student_id).strip()
            ]

        elif student_name:
            matches = students_df[
                students_df["student_name"]
                .astype(str)
                .str.contains(
                    str(student_name),
                    case=False,
                    na=False
                )
            ]

        else:
            return json.dumps({"error": "Provide student name or ID."})

        if matches.empty:
            return json.dumps({"error": "Student not found."})

        return matches.to_json(orient="records", date_format="iso")

    # ---------------------------------------------
    # STUDENT ACADEMIC SUMMARY
    # ---------------------------------------------

    def student_academic_summary(student_id):

        student_rows = students_df[
            students_df["student_id"].astype(str)
            == str(student_id)
        ]

        if student_rows.empty:

            return json.dumps({
                "error": "Student not found."
            })

        student = student_rows.iloc[0]

        sid = str(student["student_id"])

        att = attendance_df[
            attendance_df["student_id"].astype(str)
            == sid
        ].copy()

        weighted_attendance, _ = (
            calculate_weighted_attendance(att)
        )

        if len(att) > 0:

            statuses = (
                att["status"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            raw_attendance = (
                (
                    (statuses == "present").sum()
                    + 0.5 * (statuses == "late").sum()
                )
                / len(att)
                * 100
            )

        else:

            raw_attendance = 0.0

        assessments = assessments_df[
            assessments_df["student_id"].astype(str)
            == sid
        ].copy()

        component_weights = {
            "quiz": 0.20,
            "assignment": 0.30,
            "project": 0.25,
            "exam": 0.25
        }

        component_averages = {}

        assessment_types = (
            assessments["type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        for component in component_weights:

            if component == "quiz":

                selected = assessments[
                    assessment_types.str.contains(
                        "quiz",
                        na=False
                    )
                ]

            else:

                selected = assessments[
                    assessment_types == component
                ]

            total_max = pd.to_numeric(
                selected["max_score"],
                errors="coerce"
            ).sum()

            total_scored = pd.to_numeric(
                selected["scored"],
                errors="coerce"
            ).sum()

            component_averages[component] = (
                total_scored / total_max * 100
                if total_max > 0
                else 0.0
            )

        grade = sum(
            component_averages[key] * weight
            for key, weight in component_weights.items()
        )

        course_rows = courses_df[
            courses_df["course_id"].astype(str)
            == str(student["course_enrolled"])
        ]

        total_modules = (
            float(course_rows.iloc[0]["total_modules"])
            if not course_rows.empty
            else 0
        )

        enrollment_date = pd.to_datetime(
            student["enrollment_date"]
        )

        weeks_since_enrollment = (
            REFERENCE_DATE - enrollment_date
        ).days / 7

        expected_module = (
            min(weeks_since_enrollment / 3, total_modules)
            if total_modules > 0
            else 0
        )

        current_module = float(
            student["current_module"]
        )

        if (
            current_module >= total_modules
            and total_modules > 0
        ):

            progress_status = "Completed"

        elif current_module >= expected_module:

            progress_status = "On Track"

        elif current_module >= expected_module - 1:

            progress_status = "Slightly Behind"

        elif current_module >= expected_module - 3:

            progress_status = "Behind"

        else:

            progress_status = "At Risk"

        completion_pct = (
            current_module / total_modules * 100
            if total_modules > 0
            else 0
        )

        if grade < 50 or weighted_attendance < 60:

            alert = "Red"

        elif grade < 65 or weighted_attendance < 70:

            alert = "Yellow"

        elif grade >= 75 and weighted_attendance >= 75:

            alert = "Green"

        else:

            alert = "Orange"

        result = {

            "student_id": sid,

            "student_name": str(
                student["student_name"]
            ),

            "batch": str(student["batch"]),

            "course": str(
                student["course_enrolled"]
            ),

            "current_module": current_module,

            "expected_module": round(
                expected_module, 2
            ),

            "progress_status": progress_status,

            "completion_pct": round(
                completion_pct, 2
            ),

            "component_averages": {
                key: round(value, 2)
                for key, value in component_averages.items()
            },

            "overall_grade": round(grade, 2),

            "raw_attendance_pct": round(
                raw_attendance, 2
            ),

            "weighted_attendance_pct": round(
                weighted_attendance, 2
            ),

            "alert": alert
        }

        return json.dumps(result)

    # ---------------------------------------------
    # REQUIRED PRESENT LECTURE SESSIONS
    # ---------------------------------------------

    def required_present_sessions(
        student_id,
        target_pct=75
    ):

        rows = attendance_df[
            attendance_df["student_id"].astype(str)
            == str(student_id)
        ].copy()

        if rows.empty:

            return json.dumps({
                "error": "No attendance records found."
            })

        if "session_type" not in rows.columns:

            return json.dumps({
                "error": "Session type column missing."
            })

        current_pct, weighted_total = (
            calculate_weighted_attendance(rows)
        )

        statuses = (
            rows["status"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        session_types = (
            rows["session_type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        weights = session_types.map(
            lambda value: 1.5 if value == "lab" else 1.0
        )

        credits = statuses.map(
            lambda value:
            1.0 if value == "present"
            else 0.5 if value == "late"
            else 0.0
        )

        weighted_present = float(
            (weights * credits).sum()
        )

        target = float(target_pct) / 100

        if not 0 < target < 1:

            return json.dumps({
                "error": "Invalid target percentage."
            })

        if current_pct >= target_pct:

            additional = 0

        else:

            additional = math.ceil(
                (
                    target * weighted_total
                    - weighted_present
                )
                / (1 - target)
            )

            additional = max(0, additional)

        return json.dumps({

            "student_id": str(student_id),

            "current_weighted_attendance_pct": round(
                current_pct, 2
            ),

            "target_attendance_pct": float(
                target_pct
            ),

            "weighted_total": round(
                weighted_total, 2
            ),

            "weighted_present": round(
                weighted_present, 2
            ),

            "additional_present_lecture_sessions": int(
                additional
            )
        })

    # ---------------------------------------------
    # BATCH SUMMARY
    # ---------------------------------------------

    def batch_summary(batch_name):

        batch_students = students_df[
            students_df["batch"].astype(str).str.lower()
            == str(batch_name).lower()
        ].copy()

        if batch_students.empty:

            return json.dumps({
                "error": "Batch not found."
            })

        merged = batch_students.merge(
            courses_df,
            left_on="course_enrolled",
            right_on="course_id",
            how="left"
        )

        merged["completion_pct"] = (
            pd.to_numeric(
                merged["current_module"],
                errors="coerce"
            )
            / pd.to_numeric(
                merged["total_modules"],
                errors="coerce"
            )
        ) * 100

        return json.dumps({

            "batch": batch_name,

            "total_students": int(
                len(merged)
            ),

            "average_completion_pct": round(
                merged["completion_pct"].mean(),
                2
            )
        })

    # ---------------------------------------------
    # COURSE SUMMARY
    # ---------------------------------------------

    def course_summary(course_id):

        course_students = students_df[
            students_df["course_enrolled"].astype(str)
            == str(course_id)
        ].copy()

        if course_students.empty:

            return json.dumps({
                "error": "Course not found or no students enrolled."
            })

        course_rows = courses_df[
            courses_df["course_id"].astype(str)
            == str(course_id)
        ]

        if course_rows.empty:

            return json.dumps({
                "error": "Course details not found."
            })

        total_modules = float(
            course_rows.iloc[0]["total_modules"]
        )

        average_completion = (
            pd.to_numeric(
                course_students["current_module"],
                errors="coerce"
            ) / total_modules * 100
        ).mean()

        return json.dumps({

            "course_id": str(course_id),

            "course_name": str(
                course_rows.iloc[0]["course_name"]
            ),

            "total_students": int(
                len(course_students)
            ),

            "average_completion_pct": round(
                average_completion, 2
            )
        })

    # ---------------------------------------------
    # FUNCTION REGISTRY
    # ---------------------------------------------

    functions = {

        "find_student": find_student,

        "student_academic_summary":
            student_academic_summary,

        "required_present_sessions":
            required_present_sessions,

        "batch_summary": batch_summary,

        "course_summary": course_summary
    }

    # ---------------------------------------------
    # TOOL SCHEMAS
    # ---------------------------------------------

    schemas = [

        {
            "type": "function",
            "function": {
                "name": "find_student",
                "description": "Find a student using their name or student ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_name": {
                            "type": "string",
                            "description": "Student's full or partial name."
                        },
                        "student_id": {
                            "type": "string",
                            "description": "Student ID, for example STU0545."
                        }
                    }
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "student_academic_summary",
                "description": "Get a student's academic summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string"
                        }
                    },
                    "required": ["student_id"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "required_present_sessions",
                "description": "Calculate additional present lecture sessions required to reach a target weighted attendance percentage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string"
                        },
                        "target_pct": {
                            "type": "number"
                        }
                    },
                    "required": ["student_id"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "batch_summary",
                "description": "Calculate average completion for a batch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "batch_name": {
                            "type": "string"
                        }
                    },
                    "required": ["batch_name"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "course_summary",
                "description": "Calculate average completion for a course.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_id": {
                            "type": "string"
                        }
                    },
                    "required": ["course_id"]
                }
            }
        }
    ]

    return functions, schemas


# =====================================================
# AGENT SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
You are an intelligent Agentic RAG assistant for SkillBridge EdTech.

Use structured student, course, assessment and attendance datasets,
retrieved institutional policy documents and available tools.

IMPORTANT INSTRUCTIONS:
1. Always use tools when structured dataset information is required.
2. For student name or ID lookup, use find_student.
3. For a student's academic summary or progress status, use student_academic_summary.
4. For batch questions, use batch_summary.
5. For course questions, use course_summary.
6. For additional present lecture sessions, use required_present_sessions.
7. If the question says 75%, use target_pct=75.
8. Follow retrieved policy context.
9. Never invent records, scores, attendance or policy rules.
10. Reference date: January 15, 2025.

OUTPUT FORMAT:
A. For numeric questions asking for a number, percentage, count, score,
or calculation, return ONLY this JSON with exactly four keys:
{
  "answerable": 1,
  "lookup_value": 0.0,
  "rule_value": 0.0,
  "final_answer": 0.0
}
If the numeric question cannot be answered, return:
{
  "answerable": 0,
  "lookup_value": -1,
  "rule_value": -1,
  "final_answer": -1
}

B. For descriptive questions asking for names, progress status, student
information, explanations or other text, return ONLY:
{
  "answer_type": "text",
  "final_answer": "Your answer here"
}
Do not convert names or text into numbers.
Do not include explanations outside the required JSON.
"""


# =====================================================
# AGENT PIPELINE
# =====================================================

def run_agent(
    query,
    policy_context,
    tool_functions,
    openai_tools,
    pool,
    max_turns=6
):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": f"""
Question:
{query}

Retrieved Policy Context:
{policy_context}

Use the policy context and available tools to answer.
Follow the correct output format from the system instructions.
"""
        }
    ]

    for turn in range(max_turns):

        response = call_llm(
            messages,
            openai_tools,
            pool
        )

        assistant_message = (
            response.choices[0].message
        )

        if not assistant_message.tool_calls:

            return assistant_message.content or ""

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

                if tool_name in tool_functions:

                    tool_result = tool_functions[
                        tool_name
                    ](**arguments)

                else:

                    tool_result = json.dumps({
                        "error": f"Unknown tool: {tool_name}"
                    })

            except Exception as e:

                tool_result = json.dumps({
                    "error": str(e)
                })

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": str(tool_result)
            })

    # Final response without tools.
    messages.append({

        "role": "user",

        "content": (
            "Provide the final answer using the correct output format "
            "for the user question. Do not call tools."
        )
    })

    final_response = call_llm(
        messages,
        None,
        pool
    )

    return (
        final_response.choices[0].message.content
        or ""
    )


# =====================================================
# RESPONSE PARSER
# =====================================================

def parse_response(response_text):

    if not response_text:
        raise ValueError("LLM returned an empty response.")

    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        return {
            "answer_type": "text",
            "final_answer": cleaned
        }

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "answer_type": "text",
            "final_answer": cleaned
        }

    # Keep the original Kaggle numeric four-field format unchanged.
    required_keys = [
        "answerable",
        "lookup_value",
        "rule_value",
        "final_answer"
    ]

    if all(key in data for key in required_keys):
        result = {
            "answerable": int(data["answerable"]),
            "lookup_value": float(data["lookup_value"]),
            "rule_value": float(data["rule_value"]),
            "final_answer": float(data["final_answer"])
        }

        if result["answerable"] == 0:
            result["lookup_value"] = -1
            result["rule_value"] = -1
            result["final_answer"] = -1

        return result

    # Text answers are returned as text and are never cast to float.
    if data.get("answer_type") == "text":
        return {
            "answer_type": "text",
            "final_answer": str(data.get("final_answer", ""))
        }

    # Graceful fallback for valid JSON that does not match either schema.
    return {
        "answer_type": "text",
        "final_answer": str(data.get("final_answer", data))
    }

# =====================================================
# INITIALIZE BACKEND
# =====================================================

@st.cache_resource
def initialize_backend():

    groq_keys, google_key = get_api_keys()

    pool = KeyPool(groq_keys)

    gemini_client = genai.Client(
        api_key=google_key
    )

    (
        courses_df,
        mentors_df,
        students_df,
        assessments_df,
        attendance_df,
        policy_docs
    ) = load_data()

    policy_chunks, faiss_index = create_rag(
        gemini_client,
        policy_docs
    )

    tool_functions, openai_tools = create_agent_tools(
        courses_df,
        students_df,
        assessments_df,
        attendance_df
    )

    return {

        "pool": pool,

        "gemini_client": gemini_client,

        "policy_chunks": policy_chunks,

        "faiss_index": faiss_index,

        "tool_functions": tool_functions,

        "openai_tools": openai_tools,

        "students_df": students_df,

        "courses_df": courses_df
    }


# =====================================================
# PUBLIC CHAT FUNCTION
# =====================================================

def ask_agent(query):

    if not query or not query.strip():

        raise ValueError("Please enter a question.")

    backend = initialize_backend()

    policies = retrieve_policies(
        query,
        backend["gemini_client"],
        backend["policy_chunks"],
        backend["faiss_index"]
    )

    policy_context = "\n\n".join([

        f"Source: {p['source']}\n{p['text']}"

        for p in policies
    ])

    raw_response = run_agent(
        query,
        policy_context,
        backend["tool_functions"],
        backend["openai_tools"],
        backend["pool"]
    )

    return parse_response(raw_response)
