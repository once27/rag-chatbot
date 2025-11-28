from sentence_transformers import SentenceTransformer
import os

os.makedirs('./models', exist_ok=True) # create directory

m = SentenceTransformer('all-MiniLM-L6-v2') # load the model

m.save('./models/all-MiniLM-L6-v2') # save the model to the folder
print("Saved model to ./models/all-MiniLM-L6-v2")
