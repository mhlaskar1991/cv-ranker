import json
import re
from pypdf import PdfReader


def load_file(path):

    # Handle PDF resumes
    if path.lower().endswith(".pdf"):

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text

    # Handle text files
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def extract_skills_llm(text, tokenizer, model):

#     prompt = f"""
# Extract the key technical skills from the following text.
#
# The text may be a job description or a candidate CV.
#
# Return ONLY a comma-separated list of skills.
#
# Example:
# Java, Spring Boot, REST APIs, SQL
#
# Text:
# {text}
# """

    prompt = f"""
    Extract ONLY short technical skills from the following text.

    Return a comma-separated list.
    Each skill must be 1–3 words.

    Example:
    Java, Spring Boot, REST APIs, SQL, Docker

    Text:
    {text}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = re.sub(r"skills?:", "", result.lower())
    result = re.sub(r"[.\n]", "", result)

    skills = [s.strip() for s in result.split(",") if s.strip()]

    return skills


def skill_overlap(job_skills, cv_text, tokenizer, model):

    cv_skills = extract_skills_llm(cv_text, tokenizer, model)

    overlap = 0

    for js in job_skills:
        for cs in cv_skills:
            if js in cs or cs in js:
                overlap += 1

    return overlap