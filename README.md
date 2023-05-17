# Google_BARD_PALM2
Getting Started with the Vertex AI PaLM API &amp; Python SDK. Large language models (LLMs) are deep learning models trained on massive datasets of text.

# PALM
PaLM, PaLM 2 is Google's next generation large language model that builds on Google’s legacy of breakthrough research in machine learning and responsible AI. PaLM 2 excels at tasks like advanced reasoning, translation, and code generation because of how it was built.

PaLM 2 excels at advanced reasoning tasks, including code and math, classification and question answering, translation and multilingual proficiency, and natural language generation better than our previous state-of-the-art LLMs, including PaLM. It can accomplish these tasks because of the way it was built – bringing together compute-optimal scaling, an improved dataset mixture, and model architecture improvements.

PaLM 2 is grounded in Google’s approach to building and deploying AI responsibly. It was evaluated rigorously for its potential harms and biases, capabilities and downstream uses in research and in-product applications. It’s being used in other state-of-the-art models, like Med-PaLM 2 and Sec-PaLM, and is powering generative AI features and tools at Google, like Bard and the PaLM API.

PaLM is pre-trained on a wide range of text data using an unsupervised learning approach, without any specific task. During this pre-training process, PaLM learns to predict the next word in a sentence, given the preceding words. This enables the model to generate coherent, fluent text resembling human writing. This large size enables it to learn complex patterns and relationships in language and generate high-quality text for various applications. This is why models like PaLM are referred to as "foundational models."

# Objectives
In this tutorial, you will learn how to use PaLM API with the Python SDK and explore its various parameters.

By the end of the notebook, you should be able to understand various nuances of generative model parameters like temperature, top_k, top_p, and how each parameter affects the results.

The steps performed include:

Installing the Python SDK
Using Vertex AI PaLM API
Text generation model with text-bison@001
Understanding model parameters (temperature, max_output_token, top_k, top_p)
Chat model with chat-bison@001
Embeddings model with textembedding-gecko@001

# Install Vertex AI SDK
!pip install google-cloud-aiplatform --upgrade --user

# Vertex AI PaLM API models
The Vertex AI PaLM API enables you to test, customize, and deploy instances of Google’s large language models (LLM) called as PaLM, so that you can leverage the capabilities of PaLM in your applications.

# Available models
The Vertex AI PaLM API currently supports three models:

text-bison@001 : Fine-tuned to follow natural language instructions and is suitable for a variety of language tasks.
chat-bison@001 : Fine-tuned for multi-turn conversation use cases like building a chatbot.
textembedding-gecko@001 : Returns model embeddings for text inputs.


# Text generation with text-bison@001
The text generation model from PaLM API that you will use in this notebook is text-bison@001. It is fine-tuned to follow natural language instructions and is suitable for a variety of language tasks, such as:

Classification
Sentiment analysis
Entity extraction
Extractive question-answering
Summarization
Re-writing text in a different style
Ad copy generation
Concept ideation
Concept simplification

# Load model
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#Prompt design
Prompt design is the process of creating prompts that elicit the desired response from a language model. Prompt design is an important part of using language models because it allows non-specialists to control the output of the model with minimal overhead. By carefully crafting the prompts, you can nudge the model to generate a desired result. Prompt design can be an efficient way to experiment with adapting an LLM for a specific use case. The iterative process of repeatedly updating prompts and assessing the model’s responses is sometimes called prompt engineering.
