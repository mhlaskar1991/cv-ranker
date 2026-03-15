def extract_skills(text, tokenizer, model, max_tokens):

    prompt = f"""
Extract ONLY the technical skills that appear explicitly in the resume.

Rules:
- Do NOT guess skills
- Do NOT add skills not present in the text
- Return a comma separated list
- If no skills appear return: None

Example:

Resume:
Python developer with experience in Django and REST APIs
Answer:
Python, Django, REST APIs

Resume:
{text}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    skills = [
        s.strip().lower()
        for s in result.split(",")
        if s.strip() and s.strip().lower() != "none"
    ]

    return skills


def extract_education(text, tokenizer, model, max_tokens):

    prompt = f"""
Extract the highest education qualification from the resume.

Rules:
- Use ONLY information present in the resume
- Do NOT guess or invent degrees
- If no education is mentioned return: None

Examples:

Resume:
B.Tech Computer Science from IIT Delhi
Answer:
B.Tech Computer Science

Resume:
{text}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result.strip()


def extract_experience(text, tokenizer, model, max_tokens):

    prompt = f"""
Extract the candidate's professional experience from the resume.

Return ONLY a short phrase containing:
- the number of years
- the job role

Rules:
- The answer MUST contain the number of years
- Do not invent information
- Copy wording from the resume when possible
- If no experience is found return: None

Examples:

Resume:
Software developer with 3 years experience in Python and Django
Answer:
3 years software developer

Resume:
Experience:
7 years as Backend Software Engineer
Answer:
7 years Backend Software Engineer

Resume:
{text}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result.strip()


def parse_cv(cv_text, tokenizer, model, max_tokens):

    skills = extract_skills(cv_text, tokenizer, model, max_tokens)

    education = extract_education(cv_text, tokenizer, model, max_tokens)

    experience = extract_experience(cv_text, tokenizer, model, max_tokens)

    return {
        "skills": skills,
        "education": education,
        "experience": experience
    }