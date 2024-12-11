# Step 5: Load YAML File and Parse Configurations
from typing import List, Dict, Type
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from ragbuilder.generation.config import GenerationOptionsConfig, GenerationConfig
from ragbuilder.generation.components import LLM_MAP, LLMConfig
from ragbuilder.generation.prompt_templates import load_prompts
from ragbuilder.generation.sample_retriever import sample_retriever
from ragbuilder.generation.evaluation import RAGASEvaluator

class SystemPromptGenerator:
    def __init__(self, config: GenerationOptionsConfig, evaluator_class: Type):
        self.llms = []  # List to store instantiated LLMs
        self.config = config
        print("printing config",config) 
        print("printing config",config)
        # self.evaluator = evaluator_class() 
        self.retriever=sample_retriever
        if config.prompt_template_path:
            self.prompt_templates = load_prompts(config.prompt_template_path)
        else:
            self.prompt_templates = load_prompts()
        # for llm_config in config.llms:
        #     llm_class = LLM_MAP[llm_config.type]  # Get the corresponding LLM class)
    def _build_trial_config(self) -> List[GenerationConfig]:
            """
            Build a list of GenerationConfig objects from the provided GenerationOptionsConfig.

            Args:
                options_config (GenerationOptionsConfig): The input configuration for trial generation.

            Returns:
                List[GenerationConfig]: A list of generated configurations for trials.
            """
            trial_configs = []
            for llm_config in self.config.llms:
                # llm_class = LLM_MAP[llm_config.type]  # Get the corresponding LLM class)
                llm_instance = LLMConfig(type=llm_config.type, model_kwargs=llm_config.model_kwargs)
                llm_class = LLM_MAP[llm_config.type]
                # Step 8: Instantiate the Model with the Configured Parameters
                llm = llm_class(**llm_config.model_kwargs)
                # print(llm_config.type,llm_config.model_kwargs,llm.invoke("what is the capital of France?"))
                # print(self.prompt_templates)
                for prompt_template in self.prompt_templates:
                    trial_config = GenerationConfig(
                        type=llm_config.type,  # Pass the LLMConfig instance here
                        model_kwargs=llm_config.model_kwargs,
                        # evaluator=self.evaluator,
                        # retriever=self.retriever,
                        # eval_data_set_path=self.config.eval_data_set_path,
                        prompt_template=prompt_template.template,
                        # read_local_only=self.config.read_local_only,
                        )
                    # res=self._create_pipeline(trial_config,self.retriever()).invoke("Who is Clara?")
                    # print(res)
                    trial_configs.append(trial_config)
                    break

                # print(trial_config)
                # trial_configs.append(trial_config)

            # print(trial_configs)
            # for llm_config in options_config.llms:
            #     print(llm_config)
            #     trial_config = GenerationConfig(
            #         llm_type=llm_config.type,
            #         llm_model_kwargs=llm_config.model_kwargs,
            #         evaluator=options_config.evaluator,
            #         retriever=options_config.retriever,
            #         eval_data_set_path=options_config.eval_data_set_path,
            #         prompt_template_path=options_config.prompt_template_path,
            #         read_local_only=options_config.read_local_only,
            #     )
            #     trial_configs.append(trial_config)
            return trial_configs
    def _create_pipeline(self, trial_config: GenerationConfig, retriever: RunnableParallel):
        try:
            def format_docs(docs):
                return "\n".join(doc.page_content for doc in docs)

            # Prompt setup
            llm_class = LLM_MAP[trial_config.type]
                # Step 8: Instantiate the Model with the Configured Parameters
            llm = llm_class(**trial_config.model_kwargs)
            prompt_template = trial_config.prompt_template
            print('prompt_template',prompt_template)
            print("testing retriever\n",retriever.invoke("Who is Clara?"))
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_template),
                    ("user", "{question}"),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                ]
            )

            # RAG Chain setup
            rag_chain = (
                RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"])
            )
            print("rag_pipeline completed")
            return rag_chain
        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return None
    def optimize(self):
        trial_configs = self._build_trial_config()
        print("trial_configs",trial_configs)
        pipeline=None
        results = {}
        evaluator = RAGASEvaluator()
        evaldataset=evaluator.get_eval_dataset('/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/gensimtest.csv')
        for trial_config in trial_configs:
            pipeline = self._create_pipeline(trial_config,self.retriever())
            for entry in evaldataset:
                question = entry.get("question", "")
                result=pipeline.invoke(question)
                results[trial_config.prompt_template] = []
                results[trial_config.prompt_template].append({
                        "prompt_key": trial_config.prompt_template,
                        "prompt": trial_config.prompt_template,
                        "question": question,
                        "answer": result.get("answer", "Error"),
                        "context": result.get("context", "Error"),
                        "ground_truth": entry.get("ground_truth", ""),
                        "config": trial_config.dict(),
                    })
                break
        output_data = []
        for prompt_key, prompt_results in results.items():
            output_data.extend(prompt_results)

        # Convert to a Dataset directly
        from datasets import Dataset
        results_dataset = Dataset.from_list(output_data)

        # Optionally clean up or format the dataset
        if "context" in results_dataset.column_names:
            results_dataset = results_dataset.map(
                lambda x: {
                    **x,
                    "contexts": eval(x["context"]) if isinstance(x["context"], str) and x["context"].startswith("[") else [x["context"]],
                }
            )

        print("test_prompt completed")
        # return results_dataset
        eval_results=evaluator.evaluate(results_dataset)
        # print('eval_results',GenerationConfig(**eval_results['config'][0]))
        final_results=self.calculate_metrics(eval_results)
        return final_results
    # def calculate_metrics(self, result):
    #     # Convert the results to a pandas DataFrame
    #     print("writing config",result['config'])
    #     results_df = result.to_pandas()
        
    #     # Calculate average correctness per prompt key
    #     average_correctness = results_df.groupby(['prompt_key','prompt'])['answer_correctness',].mean().reset_index()
    #     average_correctness.columns = ['prompt_key', "prompt", 'average_correctness']
        
    #     # Save the average correctness to a CSV file
    #     average_correctness.to_csv('rag_average_correctness.csv', index=False)
    #     print("The average correctness results have been saved to 'rag_average_correctness.csv'")
        
    #     # Find the row with the highest average correctness
    #     best_prompt_row = average_correctness.loc[average_correctness['average_correctness'].idxmax()]
        
    #     # Extract prompt_key, prompt, and average_correctness
    #     prompt_key = best_prompt_row['prompt_key']
    #     prompt = best_prompt_row['prompt']
    #     max_average_correctness = best_prompt_row['average_correctness']
    #     config=best_prompt_row['config']
    #     return prompt_key, prompt, max_average_correctness,config
    #     return config
    def calculate_metrics(self, result):
    # Convert the results to a pandas DataFrame
        results_df = result.to_pandas()

        # Group by `prompt_key` and calculate average correctness, while retaining `prompt` and `config`
        grouped_results = (
            results_df.groupby('prompt_key')
            .agg(
                prompt=('prompt', 'first'),  # Take the first prompt for each prompt_key
                config=('config', 'first'),  # Take the first config for each prompt_key
                average_correctness=('answer_correctness', 'mean')  # Calculate average correctness
            )
            .reset_index()
        )

        # Save the results to a CSV file
        grouped_results.to_csv('rag_average_correctness.csv', index=False)
        print("The average correctness results have been saved to 'rag_average_correctness.csv'")

        # Find the row with the highest average correctness
        best_prompt_row = grouped_results.loc[grouped_results['average_correctness'].idxmax()]

        # Extract prompt_key, prompt, average_correctness, and config
        prompt_key = best_prompt_row['prompt_key']
        prompt = best_prompt_row['prompt']
        max_average_correctness = best_prompt_row['average_correctness']
        config = best_prompt_row['config']

        return prompt_key, prompt, max_average_correctness, config

        # return prompt_key, prompt, max_average_correctness