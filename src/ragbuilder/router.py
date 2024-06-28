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
from ragbuilder.langchain_module.rag.mergerag import mergerag as mergerag
# from ragbuilder.langchain_module.rag.getCode import getCodeFromFile as getCode
# from ragbuilder.langchain_module.rag.getTemplate import getTemplate as getTemplate
def router(kwargs):
    # Combination Configs
    if kwargs['framework'] == 'langchain' and kwargs.get('ragtype',None) =='config':
        code_string = LangchainCodeGen(**kwargs)
    #Byor file
    if kwargs['framework'] == 'langchain' and kwargs.get('ragtype',None) =='byor':
        code_string = getCode(**kwargs)
    # SOT Templates in a config
    if kwargs['framework'] == 'langchain' and kwargs.get('ragtype',None) =='templateConfig':
        code_string = mergerag(**kwargs)
    # SOT Templates in a file as complete code. SOT templates that cannot be done using config e.g hyde, query fusion, Subquestion etc
    if kwargs['framework'] == 'langchain' and kwargs.get('ragtype',None) =='template':
        code_string = getTemplate(**kwargs)
    return code_string