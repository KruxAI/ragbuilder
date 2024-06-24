
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
def getResponseSynthesizer(**kwargs):
    synthesize_args = kwargs['synthesize_args']
    if synthesize_args['response_mode'] == "refine":
        response_mode=ResponseMode.REFINE
    elif synthesize_args['response_mode'] == "compact":
        response_mode=ResponseMode.COMPACT
    elif synthesize_args['response_mode'] == "tree_summarize":
        response_mode=ResponseMode.TREE_SUMMARIZE
    elif synthesize_args['response_mode'] == "simple_summarize":
        response_mode=ResponseMode.SIMPLE_SUMMARIZE
    elif synthesize_args['response_mode'] == "accumulate":
        response_mode=ResponseMode.ACCUMULATE
    elif synthesize_args['response_mode'] == "compact_accumulate":
        response_mode  = ResponseMode.COMPACT_ACCUMULATE
    else :
        response_mode = None
    response_synthesizer = get_response_synthesizer(response_mode=response_mode,structured_answer_filtering=synthesize_args['structured_answer_filtering'],verbose=True)
    return response_synthesizer