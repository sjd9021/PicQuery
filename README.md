# PicQuery
End to End AI image parser which allows user to search and query images using descriptions and prompts. Used NPL, LLM and Diffusors to implement the above.

The project aims to develop a software application that enables users to search for images in their gallery by inputting image descriptions. The process involves generating text-based descriptions for images using a language model, creating text embeddings, and performing retrieval-based search using user queries. The main components of the project include:

1. Image Description Generation: The software generates textual descriptions for images using an image analysis model. This involves extracting image tags and converting them into coherent English descriptions.
2. Text Embedding Creation: The project creates a vector database for text embeddings based on the generated image descriptions. The database enables efficient retrieval of relevant descriptions.
3. User Query and Retrieval: Users can input queries in the form of image descriptions. The system retrieves relevant image descriptions from the vector database based on the user's query.

Skills and Technologies Used:

- Programming Languages: Python
- Libraries and Frameworks: requests, langchain, langchain.embeddings.openai, langchain.vectorstores, langchain.chat_models, langchain.prompts, langchain.memory, langchain.chains, langchain.document_loaders, dotenv
- APIs and Services: OpenAI API, Azure Chat API
- Machine Learning: Language models, text embeddings
- Image Processing: Image analysis, image tags extraction
- Version Control: Git
