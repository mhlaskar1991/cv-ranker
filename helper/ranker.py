import re

def score_cv(job_description, parsed_cv, tokenizer, model, max_tokens):

    # Convert parsed CV dictionary into readable profile
    candidate_profile = f"""
Skills: {', '.join(parsed_cv.get('skills', []))}
Education: {parsed_cv.get('education', 'None')}
Experience: {parsed_cv.get('experience', 'None')}
"""

    prompt = f"""
You are an AI recruitment scoring system.

Evaluate the candidate for the job.

Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Scoring Rules:
10 = perfect match
8 = strong match
6 = moderate match
4 = weak match
2 = poor match
0 = irrelevant

Return ONLY one integer from this set:
0, 2, 4, 6, 8, 10

Do not return ranges or explanations.

Score:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model output:", result)

    # Extract first number from output
    match = re.search(r"\d+", result)

    if match:
        return float(match.group())
    else:
        return 0