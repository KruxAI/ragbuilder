# Step 5: Load YAML File and Parse Configurations
from typing import List, Dict, Type
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from operator import itemgetter
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from ragbuilder.config import LogConfig, GenerationOptionsConfig, GenerationConfig
from ragbuilder.config.generation import LLM_MAP
from ragbuilder.config.generation import LLMConfig
from ragbuilder.generation.prompt_templates import load_prompts
# from ragbuilder.generation.sample_retriever import sample_retriever
from ragbuilder.generation.evaluation import Evaluator, RAGASEvaluator
from typing import List, Dict, Any, Optional
# from ragbuilder.config.components import EvaluatorType
from importlib import import_module
from ragbuilder.core.exceptions import DependencyError
from ragbuilder.core import setup_rich_logging, console
import logging
from ragbuilder.core.callbacks import DBLoggerCallback  # Import the callback class

class SystemPromptGenerator:
    def __init__(
        self, 
        config: GenerationOptionsConfig, 
        evaluator: Evaluator, 
        retriever: Optional[Any] = None, 
        verbose: bool = False,
        callback=None  # Add callback parameter
    ):
        self.logger = logging.getLogger("ragbuilder.generation.optimization")
        self.llms = []  # List to store instantiated LLMs
        self.config = config
        self.evaluator = evaluator
        self.eval_data_set_path = config.eval_data_set_path
        self.verbose = verbose
        self.callbacks = []  # Initialize callbacks list

        # Initialize DBLoggerCallback if database logging is enabled
        if config.database_logging:
            try:
                self.logger.info(f"{config.optimization.study_name},study_name")
                db_callback = DBLoggerCallback(
                    study_name=config.optimization.study_name,
                    config=config,
                    module_type='generation'
                )
                self.callbacks.append(db_callback)
                self.logger.debug("Database logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database logging: {e}")

        if callback:
            self.callbacks.append(callback)

        self.logger.debug("Loading Prompts")
        self.local_prompt_template_path = config.local_prompt_template_path
        self.read_local_only = config.read_local_only
        
        self.logger.debug(f"Initializing with config: {config}")
        self.logger.debug("Loading Retriever")
        self.retriever = retriever
        if self.retriever is None:
            raise DependencyError("Retriever Not set")
        self.prompt_templates = load_prompts(
            config.prompt_template_path, 
            config.local_prompt_template_path, 
            config.read_local_only
        )
        self.logger.debug(f"Loaded prompt templates: {len(self.prompt_templates)}")
        
    def _build_trial_config(self) -> List[GenerationConfig]:
        trial_configs = []
        self.logger.debug("Building trial configs")
        for llm_config in self.config.llms:
            for prompt_template in self.prompt_templates:
                trial_config = GenerationConfig(
                    type=llm_config.type,
                    model_kwargs=llm_config.model_kwargs,
                    eval_data_set_path=self.eval_data_set_path,
                    prompt_template=prompt_template[1].template,
                    prompt_key=prompt_template[0]
                )
                trial_configs.append(trial_config)
                # break #REMOVE
        return trial_configs
    
    def optimize(self):
        console.rule("[heading]Starting Generation Optimization[/heading]")
        
        trial_configs = self._build_trial_config()
        n_trials = len(trial_configs)
        self.logger.info(f"Generated {n_trials} trial configurations")
        
        pipeline = None
        results = {}
        
        evaldataset = self.evaluator.get_eval_dataset(self.eval_data_set_path)
        self.logger.debug(f"Loaded evaluation dataset with {len(evaldataset)} entries")

        for i, trial_config in enumerate(trial_configs):
            console.print(f"[heading]Trial {i}/{n_trials-1}[/heading]")
            if self.verbose:
                console.print(f"Running trial {i} with prompt template: {trial_config.prompt_template}")
        
            self.logger.info(f"Creating pipeline for trial {i}")
            pipeline = create_pipeline(trial_config, self.retriever)
            
            self.logger.info(f"Preparing eval dataset for trial {i}")
            for i,entry in enumerate(evaldataset):
                question_id = i
                question = entry.get("question", "")
                result = pipeline.invoke(question)
                combined_key = f"{trial_config.prompt_key}_{question_id}"  # Combine prompt_key and question_id
                results[combined_key] = []
                results[combined_key].append({
                    "prompt_key": trial_config.prompt_key,
                    "prompt": trial_config.prompt_template,
                    "question_id": question_id,
                    "question": question,
                    "answer": result.get("answer", "Error"),
                    "context": result.get("context", "Error"),
                    "ground_truth": entry.get("ground_truth", ""),
                    "config": trial_config.model_dump(),
                })
        # Convert results to Dataset
        from datasets import Dataset
        results_dataset = Dataset.from_list([item for items in results.values() for item in items])

        if "context" in results_dataset.column_names:
            results_dataset = results_dataset.map(
                lambda x: {
                    **x,
                    "contexts": eval(x["context"]) if isinstance(x["context"], str) and x["context"].startswith("[") else [x["context"]],
                }
            )

        self.logger.info(f"Evaluating prompt results")
        eval_results = self.evaluator.evaluate(results_dataset)
        self.logger.info(f"Calculating final prompt testing results")
        final_results = self.calculate_metrics(eval_results)
        
        console.print("[success]Optimization Complete![/success]")
        for callback in self.callbacks:
            try:
                callback(study=None,trial=None,eval_results=eval_results, final_results=final_results)
            except Exception as e:
                    self.logger.warning(f"Callback error: {e}")
        return final_results

    def calculate_metrics(self, result):
        results_df = result.to_pandas()
        grouped_results = (
            results_df.groupby('prompt_key')
            .agg(
                prompt=('prompt', 'first'),
                config=('config', 'first'),
                average_correctness=('answer_correctness', 'mean')
            )
            .reset_index()
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = 'rag_average_correctness' + timestamp + '.csv'
        grouped_results.to_csv('rag_average_correctness.csv', index=False)
        self.logger.info("Average correctness results saved to 'rag_average_correctness.csv'")

        best_prompt_row = grouped_results.loc[grouped_results['average_correctness'].idxmax()]
        prompt_key = best_prompt_row['prompt_key']
        prompt = best_prompt_row['prompt']
        max_average_correctness = best_prompt_row['average_correctness']
        config = best_prompt_row['config']
        best_pipeline = create_pipeline(GenerationConfig(**config), self.retriever)
        console.print(f"[success]Best Prompt:[/success] {prompt_key[:50]}...\n[success]Best Score:[/success] [value]{max_average_correctness}[/value]")

        return {
            "best_config": GenerationConfig(**config),
            "best_prompt": prompt,
            "best_score": max_average_correctness,
            "best_pipeline": best_pipeline
        }

def run_generation_optimization(options_config: GenerationOptionsConfig, retriever: Optional[Any] = None, log_config: Optional[LogConfig] = None) -> Dict[str, Any]:
    """Run Prompt optimization process."""
    setup_rich_logging(
        log_config.log_level if log_config else logging.INFO,
        log_config.log_file if log_config else None
    )
    evaluator = RAGASEvaluator()
    optimizer = SystemPromptGenerator(options_config, evaluator, retriever=retriever, verbose=log_config.verbose)
    return optimizer.optimize()

def create_pipeline(trial_config: GenerationConfig, retriever: RunnableParallel):
    logger = logging.getLogger("ragbuilder.generation.pipeline")
    logger.debug(f"Creating pipeline with config: {trial_config}")
    
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        llm_class = LLM_MAP[trial_config.type]()
        model_kwargs = {k: v for k, v in trial_config.model_kwargs.items() if v is not None}
        llm = llm_class(**model_kwargs)
        prompt_template = trial_config.prompt_template

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
        ])

        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(format_docs))
            .assign(answer=prompt | llm | StrOutputParser())
            .pick(["answer", "context"])
        )
        return rag_chain
    except Exception as e:
        import traceback
        logger.error(f"Pipeline creation failed: {e}")
        traceback.print_exc()
        return None