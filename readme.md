## Steps to run this project

1. **Install dependencies**:

   ```bash
   cd <repository-directory>
   pip install -r requirements.txt
   ```

2. **Create a folder data/docs in the root directory**:

   ```bash
   mkdir -p data/docs
   ```

   - Keep the downloaded pdfs in this folder

3. **Copy .env.example to .env**:

   ```bash
   cp .env.example .env

   /*Please dont't misuse my gemini API key (requesting)*/
   ```

4. **Run the storing module**:

   ```bash
   python -m storing

   /* run this in the root directory , this will take some time as it creates embeddings */
   ```

5. **Run the FastAPI server**:

   ```bash
   uvicorn app.main:app --reload
   ```

6. **Use Postman and send a POST request to `http://localhost:8000/ask`**:

   - **Body** (JSON):
     ```json
     {
       "question": "What is the minimum age to take a policy loan?s"
     }
     ```

> I hope this Works :)
