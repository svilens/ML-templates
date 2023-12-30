import torch
from build_encoder_llm import model

vocabulary = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
    "the", "a", "an", 
    "cat", "dog", "fish", "bird", "lion", "tiger", "elephant", "monkey",
    "runs", "jumps", "sleeps", "eats", "drinks",
    "fast", "slow", "big", "small", "red", "green", "blue", "yellow",
    "is", "was", "will", "can", "has", "have", "had", "do", "does",
    "I", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their"
]

# Create a word2id dictionary
word2id = {word: idx for idx, word in enumerate(vocabulary)}

# Print the dictionary
print(word2id)


# sentence similarity
def cosine_similarity(vec1: torch.tensor, vec2: torch.tensor):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))


def sentence_to_embeddings(sentence: str, model, word2id: dict):
    input_tensor = torch.tensor(
        [word2id.get(word, word2id["[UNK]"]) for word in sentence.split()], dtype=torch.long
    ).unsqueeze(0)
    embeddings = model(input_tensor, mask=None)
    return embeddings


def sentence_similarity(sentence1: str, sentence2: str, model, word2id: dict):
    embeddings1 = sentence_to_embeddings(sentence1, model, word2id)
    embeddings2 = sentence_to_embeddings(sentence2, model, word2id)

    # Compute the average embeddings of each sentence
    avg_embedding1 = torch.mean(embeddings1, dim=1)
    avg_embedding2 = torch.mean(embeddings2, dim=1)

    # Compute and return the cosine similarity
    return cosine_similarity(avg_embedding1, avg_embedding2)


sentence1 = "the cat has a blue fish"
sentence2 = "my sister's dog sleeps"

similarity = sentence_similarity(sentence1, sentence2, model, word2id)
similarity_score = similarity.item()