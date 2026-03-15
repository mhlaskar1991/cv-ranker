import argparse

from helper.utils import load_file, load_config, skill_overlap, extract_skills_llm
from helper.model_loader import load_model
from helper.cv_parser import parse_cv
from helper.ranker import score_cv
from helper.similarity import similarity_score, load_embedding_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--job", required=True)
    parser.add_argument("--cvs", nargs="+", required=True)

    args = parser.parse_args()

    config = load_config()

    parser_model_name = config["parser_model"]
    ranker_model_name = config["ranker_model"]
    embedding_model_name = config["embedding_model"]

    max_tokens = config["max_tokens"]

    llm_weight = config["scoring"]["llm_weight"]
    sim_weight = config["scoring"]["similarity_weight"]
    skill_weight = config["scoring"]["skill_weight"]

    similarity_scale = config["similarity_scale"]
    skill_multiplier = config["skill_overlap_multiplier"]

    print("Loading parser model...")
    parser_tokenizer, parser_model = load_model(parser_model_name)

    print("Loading ranker model...")
    ranker_tokenizer, ranker_model = load_model(ranker_model_name)

    print("Loading embedding model...")
    embedding_model = load_embedding_model(embedding_model_name)

    job_description = load_file(args.job)
    print("Job description: {}".format(job_description))

    # Extract job skills once
    print("Extracting job skills...")
    job_skills = extract_skills_llm(
        job_description,
        parser_tokenizer,
        parser_model
    )

    print("Job skills:", job_skills)

    results = []

    for cv_file in args.cvs:

        print(f"\nEvaluating {cv_file}")

        cv_text = load_file(cv_file)
        print("Parsing CV text:: {}".format(cv_text))

        # Step 1: Parse CV
        parsed_cv = parse_cv(
            cv_text,
            parser_tokenizer,
            parser_model,
            max_tokens
        )

        print("Parsed CV:", parsed_cv)

        # Step 2: LLM candidate scoring
        llm_score = score_cv(
            job_description,
            parsed_cv,
            ranker_tokenizer,
            ranker_model,
            max_tokens
        )

        print("LLM Score:", llm_score)

        # Step 3: Skill overlap
        overlap = skill_overlap(
            job_skills,
            cv_text,
            parser_tokenizer,
            parser_model
        )

        print("Skill Overlap:", overlap)

        # Step 4: Semantic similarity
        sim_score = similarity_score(
            job_description,
            cv_text,
            embedding_model
        )

        # Step 5: Final weighted score
        final_score = (
            llm_weight * llm_score +
            sim_weight * (sim_score * similarity_scale) +
            skill_weight * (overlap * skill_multiplier)
        )

        final_score = round(final_score, 2)

        print("Final Score:", final_score)

        results.append((cv_file, final_score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nFinal Ranking:\n")

    for i, (cv, score) in enumerate(results, 1):
        print(f"{i}. {cv} | Score: {score}")


if __name__ == "__main__":
    main()