# router receives 1 of below 
# 1. Config
# 2. Code 
# 3. template configs - -Sample as config meant for template config
# 4. Bespoke templates 

# router talks to code generator 

# router returns the code for the config or template

# 1. Reads config params and checks framework and calls correspoding code generator
# if config
#     Call Code Gen and convert to Code 
#     return code
# if code 
#     return code
# if template configs 
#     convert to Code 
#     return code
# if Bespoke templates
#     pick code from templates
#     return code
from ragbuilder.langchain_module.rag.getCode import codeGen as LangchainCodeGen
from ragbuilder.langchain_module.rag.byor import LangchainByor as LangchainByor
# from ragbuilder.langchain_module.rag.getCode import getCodeFromFile as getCode
# from ragbuilder.langchain_module.rag.getTemplate import getTemplate as getTemplate
def router(kwargs):
    # Combination Configs
    if kwargs['framework'] == 'langchain':
        code_string = LangchainCodeGen(**kwargs)
    #Byor file
    if kwargs['framework'] == 'langchain_byor':
        code_string = LangchainByor(kwargs['loader_kwargs']['input_path'])
    # SOT Templates in a file as complete code. SOT templates that cannot be done using config e.g hyde, query fusion, Subquestion etc
    return code_string

'/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/src/ragbuilder/langchain_module/rag/test55.py'