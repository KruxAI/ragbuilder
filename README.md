# RagBuilder

RagBuilder is a toolkit designed to help you create optimal Production-ready Retrieval-Augmented Generation (RAG) pipeline for your data. Ragbuilder iterates through all possibile combinations of RAG parameters (Eg: chunking strategy: semantic, character etc.; chunk size 1000, 2000 etc.), and evaluates them against a test data set, and provides a dashboard for reviewing what parameter values work best for your source data. RagBuilder also includes several pre-defined state-of-the-art RAG templates that have shown strong performance across various other datasets.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Set your OpenAI API key](#set-your-openai-api-key)
- [Quickstart Guide](#quickstart-guide)
- [License](#license)

## Features

- **Pre-defined RAG Templates**: Use state-of-the-art templates that have demonstrated strong performance across various datasets.
- **Custom RAG Configurations**: Create configurations from granular parameters like chunking strategy, chunking size, embedding models, retriever types etc.
- **Evaluation Dataset Options**: Choose to generate a synthetic test dataset or provide your own.
- **Automatic Reuse**: Automatically re-uses previously generated synthetic test data when applicable.
- **Easy-to-use Interface**: Intuitive UI to guide you through setting up, configuring, and reviewing your RAG configurations.


## Installation

Install using pip:

```
pip install ragbuilder
```

## Set your OpenAI API key

Make sure your OpenAI API key is available by setting it as an environment variable. In MacOS and Linux, this is the command:

```
export OPENAI_API_KEY=XXXXX
```

and on Windows it is

```
set OPENAI_API_KEY=XXXXX
```

## Quickstart Guide

Now, run ragbuilder on your command line:

```
ragbuilder
```

This will start the Ragbuilder Flask app and open the browser. If the browser window doesn't open automatically, go to http://localhost:8001/ in your browser to access the RagBuilder dashboard.

1. Click **New Project** to start building your RAG.
2. **Description:** Describe your use-case. Let's specify "Q&A Chatbot" as the description for our demo.
3. **Source Data:** Specify the path to your source data. This could be a URL, local directory or local file path. For the sake of our demo, let's specify the URL: https://lilianweng.github.io/posts/2023-06-23-agent/
4. **Select Ragbuilder options:** 
   - Use Pre-defined RAG Templates - When selected, this'll include pre-defined RAG configuration templates that have demonstrated strong performance across various datasets and related use-cases. These templates will be evaluated against your data, providing you with performance metrics for each pre-defined configuration.
   - Create Custom RAG Configurations - When selected, this'll generate multiple RAG configurations based on detailed parameters like chunking strategy, chunking size, embedding model, retriever type etc. This option may yield hundreds or even thousands of unique configurations to compare, offering a comprehensive performance analysis tailored to your dataset. *[Note]: This may take several minutes to complete.*
5. Next, in order to tailor your RAG configurations, you can unselect any specific options you wish to exclude (For eg: Unselecting "Chunking Strategy: Character" will exclude all RAG configurations that have the CharacterTextSplitter). For best results, you may want to leave all settings unchanged. But for our Quickstart demo, we will unselect everything except the below:
    - Chunking strategy: Markdown
    - Embedding model: text-embedding-3-large
    - Retriever: Vector DB - Similarity Search
    - Top k: 5
    - LLM: GPT-3.5 Turbo
6. Next, in Evaluation dataset options, you have the option to:
    - **Use Existing Synthetic Test Data:** If synthetic test data was previously generated for your dataset, this option will appear alongside the path of the existing test data.
    - **Generate Synthetic Test Data from My Dataset:** Create a new synthetic test dataset based on your existing data.
    - **Provide a Manually Created Test Dataset:** Use your own test dataset file (CSV format with "question" and "ground_truth" columns).
    For our demo, let's go ahead and create a synthetic test data by selecting the **Generate Synthetic Test Data**** option.
7. Before running the tool, let's review all your selections:
8. Review all the selections and click **Confirm**
9. After processing we should see the dashboard with the results.

# Environment Variables Setup for RagBuilder

This section provides instructions on setting up the environment variables required for the RagBuilder project. These variables need to be configured in a `.env` file located in the same directory where you run the `ragbuilder` command.

## Description

The environment variables are essential for authenticating and configuring various services used by the RagBuilder project. Below are the variables you need to set in your `.env` file.

### Environment Variables

- **OPENAI_API_KEY**: The API key for OpenAI services.
- **MISTRAL_API_KEY**: The API key for Mistral services.
- **ENABLE_ANALYTICS**: A boolean flag to enable or disable analytics. Set to `True` or `False`.
- **HUGGINGFACEHUB_API_TOKEN**: The API token for HuggingFace Hub.
- **COHERE_API_KEY**: The API key for Cohere services.
- **JINA_API_KEY**: The API key for Jina services.
- **SINGLESTOREDB_URL**: The connection string for SingleStoreDB, formatted as `userid:password@host:port/dbname`.
- **PINECONE_API_KEY**: The API key for Pinecone services.

### Example `.env` File

Create a file named `.env` in the directory where you will run the `ragbuilder` command and add the following content, replacing the placeholder values with your actual keys and connection string:

```env
# Environment variables for the RagBuilder project
OPENAI_API_KEY="XXXXXX"
MISTRAL_API_KEY="XXXXX" 
ENABLE_ANALYTICS="True"
HUGGINGFACEHUB_API_TOKEN="XXXXXX"
COHERE_API_KEY="XXXXXX"
JINA_API_KEY="XXXXXX"
SINGLESTOREDB_URL="userid:password@host:port/dbname"
PINECONE_API_KEY="XXXXXX"
```

### Instructions
1. Create a new file named .env in your project directory.
2. Copy the example content provided above into the .env file.
3. Replace the placeholder values with your actual API keys and connection string.
4. Save the .env file.
5. Ensure that the .env file is located in the same directory where you run the ragbuilder command to ensure the environment variables are properly loaded.

By setting up these environment variables, you enable the RagBuilder project to authenticate and interact with the necessary external services.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---