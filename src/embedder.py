from sentence_transformers import SentenceTransformer, util
import torch

class SymptomEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the embedding model. 
        'all-MiniLM-L6-v2' is a fast and balanced model for semantic similarity.
        """
        self.model = SentenceTransformer(model_name)
        
    def get_embedding(self, text: str):
        """Converts a string of text into a vector."""
        return self.model.encode(text, convert_to_tensor=True)

    def calculate_similarity(self, query_embedding, target_embeddings):
        """Calculates cosine similarity between a query and a list of targets."""
        return util.cos_sim(query_embedding, target_embeddings)

    def find_matches(self, query_text: str, disease_profiles: list, top_k: int = 5):
        """
        Takes a patient description and a list of disease/trial profiles,
        returns the top_k most similar matches with scores.
        """
        query_embedding = self.get_embedding(query_text)
        profile_texts = [p['description'] for p in disease_profiles]
        profile_embeddings = self.model.encode(profile_texts, convert_to_tensor=True)
        
        cosine_scores = self.calculate_similarity(query_embedding, profile_embeddings)[0]
        
        # Get top results
        top_results = torch.topk(cosine_scores, k=min(top_k, len(profile_texts)))
        
        matches = []
        for score, idx in zip(top_results[0], top_results[1]):
            matches.append({
                "profile": disease_profiles[idx],
                "score": float(score)
            })
            
        return matches
