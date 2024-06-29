

llm=ChatOpenAI(model='gpt-3.5-turbo')

WebBaseLoader(input_path='https://en.wikipedia.org/wiki/Python_(programming_language)')
    docs = loader.load()