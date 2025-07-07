# Semantic Book Recommender 

📌 Also available in [Spanish](README.es.md) | 📌 Disponible también en [Español](README.es.md)

This book recommendation system is developed using LLM, vector database, and Gradio for the user interface. The development steps are presented in the notebooks, from data extraction, exploration, and preprocessing to text classification, sentiment analysis, and finally deployment.

Built using Python, Gradio, LangChain, ChromaDB and OpenAIEmbeddings.

![](screenshot.jpg)

### **📚 Project demo:** 
Click here to use it: [:book: Book Recommender](https://huggingface.co/spaces/diegosruiz18/book-recommendations)  

Click the **"Restart Space"** button if the application does not appear.

## Objective

This system is designed to recommend books based on a **custom description that users enters**. Once submitted, the system returns relevant or related titles based on user's interest. Additionally, filters can be applied by category and emotional tone of the books.

## How it works?

1. First of all, user writes **description in Spanish** about the type of book that they are interested in.
2. The description is translated into English in order to match the database.
3. It is converted into a vector using **OpenAIEmbeddings**.
4. A semantic search is performed on the **ChromaDB** vector database.
5. Results are filtered by **category** and **emotional tone**.
6. Recommended books are displayed in a gallery with cover image, title, author, and description (this last one in Spanish).

### How to use it?

- Describe a book or topic that you are interested in (e.g. *a story about World War II*).
- Filter by category (fiction, nonfiction, etc).
- Sort by the emotional tone of the books (sadness, joy, suspense, etc).
- View and select any book from the galery to see more details.

![](recomendacion.jpg)

## Development

### Data preparation
- Data source: Kaggle
- Text data extraction and cleaning in ```data_exploration.ipynb```

### Vector search
- Semantic vector search and vector database construction in ```vector_search.ipynb```
- The system finds the most similar books to a natural language query (e.g., "a book about nature and animals").

### Text classification
- Text classification using zero-shoot classification in ```text_classification.ipynb```
- Books are classified as "fiction" or "nonfiction", allowing users to filter them.

### Sentiment analysis
- LLM is used to extract emotions from the text in ```sentiment_analysis.ipynb```, 
- Books are sorted by tone, such as how suspenseful, cheerful, or sad they are.

### Web Application and Deployment
- Web application built in Gradio so that users can interact with the system in ```dashboard_gradio.py```.
- Application deployed on Hugging Face.

## Technologies used

- **Python** (PyCharm)
- **Gradio** (user interface)
- **LangChain** and **Chroma** (vector database)
- **OpenAI Embeddings** (`text-embedding-ada-002` model)
- **Translation with GPT-3.5 turbo**
- **Libraries:** pandas, numpy, matplotlib, seaborn
- **Hugging Face Spaces**

All dependencies are listed in requirements.txt file.
