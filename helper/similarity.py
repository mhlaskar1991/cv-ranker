from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_embedding_model(model_name):
    model = SentenceTransformer(model_name)
    return model


def similarity_score(job_desc, cv_text, model):

    embeddings = model.encode([job_desc, cv_text])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(score)