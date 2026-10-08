1. Installed all the required dependecies for, will progess and add additional dependecies as required by this project
2. Setup a normal fastAPI server and ran it using uvicorn
3. Created a `pdf_loader` function in ingestion, which takes pdf as input and converts it to a python where each item contains a page number and the text extracted from that page
4. Created a basic `testing_pipeline.py` which is used for testing whether each step works before connecting it to FastAPI
5. Created `metadata.py` to extract basic document metadata from the file path, including
   1. Document ID
   2. Filename
   3. Department
   4. File Type
6. Created a `chunker.py` which takes the page text returned by `load_pdf()` and splits into smaller overlapping chunks (PDF Page Text -> 500 Character Chunk -> next 500 character chunk with 100 repeated character -> store page number with every chunk)
7. Created an `embedder.py` which converts chunks into vector, and also user_query into a vector, tested it in testing_pipeline
8. Created `dense.py` which compares the query_embeddings (user_query) against all chunk_embeddings (all our vectors), the embeddings are normalized, so this acts like cosine similarity
9. sorts the scores in descending order
10. Return the top k chunks



















