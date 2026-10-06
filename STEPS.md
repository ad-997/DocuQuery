1. Installed all the required dependecies for, will progess and add additional dependecies as required by this project
2. Setup a normal fastAPI server and ran it using uvicorn
3. Created a `pdf_loader` function in ingestion, which takes pdf as input and converts it to a python where each item contains a page number and the text extracted from that page
4. Created a basic `testing_pipeline.py` which is used for testing whether each step works before connecting it to FastAPI
5. Created a `chunker.py` which takes the page text returned by `load_pdf()` and splits into smaller overlapping chunks (PDF Page Text -> 500 Character Chunk -> next 500 character chunk with 100 repeated character -> store page number with every chunk)
6. 