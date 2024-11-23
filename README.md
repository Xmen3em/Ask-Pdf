<h1 align="center">
  Ask-Pdf  🤖
</h1>


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![LangChain](https://img.shields.io/badge/LangChain-blueviolet?style=for-the-badge) 
![Groq](https://img.shields.io/badge/Groq-orange?style=for-the-badge) 
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) 
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-ffcc33?style=for-the-badge&logo=huggingface&logoColor=black)

You can use the power of a Large Language Model (LLM) along with a vector database to ask questions about your own documents with an interactive GUI based on Streamlit. Currently, it works with the Groq API. You can use only PDF files.

Built with Langchain, HuggingFace, Streamlit, FAISS vector database, and Sentence Transformers. Please see the below demo-



<h1 align="center"> Getting Started 🚶 </h1>

To get your environment ready for running the code provided here, start by installing all the required dependencies (we recommend using a virtual environment):

```bash
pip install -r requirements.txt
```

### Setting Up .env

I have set up a few environment variables for ease of use. Therefore, you need to configure the necessary environment variables before proceeding to use the GUI.
add your groq API key if you want to use the same api to use the gemma2-9b-It model as your llm and also add HuggingFace api for use pretrained model for embeddings.

### Running the application through GUI 

This project uses Streamlit for frontend GUI, you can run the ```app_1.py``` script to launch the application.

```
python3 app_1.py
```
When you execute this script, the application will launch on a local server, and you can monitor the local server's output in your terminal.

<h1 align="center"> How Does it Work? 🤔</h1>

This work is based on the principle of Retrieval Augmentation Generation (RAG) which is an approach in the field of Natural Language Processing (NLP) that combines three key components: retrieval, augmentation, and generation. This approach is designed to improve the performance and capabilities of language models.

![RAG Diagram](assets\image\RAG_diagram_dark.png)

1. Documents and File Types: At the top, it shows a variety of document types (DOC, XLS, PPT, PDF) being ingested into the system, which matches your project since you're working with PDF documents as input for a retrieval system. 

2. Splitting and Vectorization: The documents are split and then stored in a Vector Database. This is relevant because in your project, you use FAISS (a vector database) to store embeddings of document chunks for efficient retrieval. Splitting documents into chunks is crucial for managing large text inputs and allowing efficient querying. 

3. Retrieval and Query Embedding: A Retriever component is present, which takes user queries, embeds them, and uses these embeddings to retrieve relevant document chunks. This process is similar to how your project uses embeddings and LangChain to retrieve answers based on the query. 

4. LLM Response Generation: The LLM (Large Language Model) receives the retrieved document chunks and generates an answer. This is in line with your project as well, where the conversational RAG (retrieval-augmented generation) process uses the retrieved content to produce a response for the user query. 

5. User Interaction: The flow starts with a User query, goes through retrieval and LLM response generation, and ends with an Answer sent back to the user. This accurately represents the conversational loop in your AskPdf project. 

## Contact
Feel free to reach out for collaborations or inquiries:

- **Email**: [abdelmoneimmohamedrehab@gmail.com](mailto:rehababdelmoneim755@gmail.com)

- **Linkedin** : [www.linkedin.com/in/abdelmoneim77](www.linkedin.com/in/abdelmoneim77)