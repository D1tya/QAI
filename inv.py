document1 = "The quick brown fox jumped over the lazy dog." 
document2 = "The lazy dog slept in the sun."
# Convert each document to lowercase and split it into words
tokens1 = document1.lower().split()
tokens2 = document2.lower().split()
print(f"Token1 : {tokens1}")
print(f"\n\nToken2 : {tokens2}")

terms = list(set(tokens1 + tokens2))
print(f"Terms : {terms}")

inverted_index = {}

# For each term, find the documents that contain it 
for term in terms:
    documents = []
    if term in tokens1: 
        documents.append("Document 1")
    if term in tokens2: 
        documents.append("Document 2")
    inverted_index[term] = documents

print(f"Inverted Index Dictionary : \n{inverted_index}")

for term, documents in inverted_index.items(): 
    print(f"{term} -> {', '.join(documents)}")