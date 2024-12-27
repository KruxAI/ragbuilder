from typing import List, Optional, Any
from operator import itemgetter
from datetime import datetime
from ragbuilder.config import LogConfig, GenerationOptionsConfig, GenerationConfig
from ragbuilder.generation.prompt_templates import load_prompts
from ragbuilder.generation.evaluation import Evaluator, RAGASEvaluator
from ragbuilder.core.exceptions import DependencyError
from ragbuilder.core import setup_rich_logging, console
import logging
from ragbuilder.core.callbacks import DBLoggerCallback
from ragbuilder.core.results import GenerationResults
from ragbuilder.core.config_store import ConfigStore
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
        self.config = config
        self.evaluator = evaluator
        self.eval_data_set_path = config.eval_data_set_path
        self.verbose = verbose
        self.callbacks = []  # Initialize callbacks list
        self.n_trials = config.optimization.n_trials
        
        # Initialize DBLoggerCallback if database logging is enabled
        if config.database_logging:
            try:
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
        counter=0
        for llm_config in self.config.llms:
            for prompt_template in self.prompt_templates: 
                if counter>=self.n_trials:
                    break
                counter+=1
                trial_config = GenerationConfig(
                    llm=llm_config,
                    prompt_template=prompt_template[1].template,
                    prompt_key=prompt_template[0]
                )
                trial_configs.append(trial_config)
            if counter>=self.n_trials:
                break
        return trial_configs
    
    def optimize(self) -> GenerationResults:
        """
        Run optimization process.
        
        Returns:
            GenerationResults containing optimization results and best pipeline
        """
        console.rule("[heading]Starting Generation Optimization[/heading]")
        
        start_time = datetime.now()
        
        trial_configs = self._build_trial_config()
        self.logger.info(f"Generated {self.n_trials} trial configurations")
        
        pipeline = None
        results = {}
        # self.logger.info(f"eval path {self.eval_data_set_path}")
        evaldataset = self.evaluator.get_eval_dataset(self.evaluator.test_dataset)
        self.logger.debug(f"Loaded evaluation dataset with {len(evaldataset)} entries")

        for i, trial_config in enumerate(trial_configs):
            console.print(f"[heading]Trial {i}/{self.n_trials-1}[/heading]")
            if self.verbose:
                console.print(f"Running trial {i} with prompt template: {trial_config.prompt_template}")
        
            self.logger.info(f"Creating pipeline for trial {i}")
            pipeline = create_pipeline(trial_config, self.retriever)
            
            self.logger.info(f"Preparing eval dataset for trial {i}")
            for i, entry in enumerate(evaldataset):
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
        
        # Create structured results object
        results = GenerationResults(
            best_config=final_results["best_config"],
            best_score=final_results["best_score"],
            best_pipeline=final_results["best_pipeline"],
            best_prompt=final_results["best_prompt"],
            n_trials=self.n_trials,
            completed_trials=len(trial_configs),
            optimization_time=(datetime.now() - start_time).total_seconds(),
            # Add performance metrics if available from eval_results
            avg_latency=None,
            error_rate=None
        )
        
        # Log results
        console.print("[success]Optimization Complete![/success]")
        console.print(f"[success]Best Score:[/success] [value]{results.best_score:.4f}[/value]")
        console.print("[success]Best Configuration:[/success]")
        console.print(results.get_config_summary(), style="value")
        
        # Call callbacks with the new results structure
        for callback in self.callbacks:
            try:
                callback(
                    study=None,
                    trial=None,
                    eval_results=eval_results,
                    final_results=results
                )
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")
                
        return results

    def calculate_metrics(self, result):
        """Calculate metrics from evaluation results."""
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
        
        # Save results to CSV for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = f'rag_average_correctness_{timestamp}.csv'
        grouped_results.to_csv(output_csv_path, index=False)
        self.logger.info(f"Average correctness results saved to '{output_csv_path}'")

        best_prompt_row = grouped_results.loc[grouped_results['average_correctness'].idxmax()]
        
        best_config = GenerationConfig(**best_prompt_row['config'])
        if best_config.llm.type is None:
            # If LLM type is None, it means we need to use the default LLM
            best_config.llm = ConfigStore().get_default_llm()
        
        return {
            "best_config": best_config,
            "best_prompt": best_prompt_row['prompt'],
            "best_score": best_prompt_row['average_correctness'],
            "best_pipeline": create_pipeline(best_config, self.retriever),
            #TODO: Add latency and error rate
            # "avg_latency": best_prompt_row.get('avg_latency'),
            # "error_rate": best_prompt_row.get('error_rate')
        }

def run_generation_optimization(
    options_config: GenerationOptionsConfig, 
    retriever: Optional[Any] = None, 
    log_config: Optional[LogConfig] = None
) -> GenerationResults:
    """
    Run Prompt optimization process.
    
    Args:
        options_config: Generation configuration options
        retriever: Optional retriever to use for generation
        log_config: Optional logging configuration
    
    Returns:
        GenerationResults containing optimization results
    """
    setup_rich_logging(
        log_config.log_level if log_config else logging.INFO,
        log_config.log_file if log_config else None
    )
    
    evaluator = RAGASEvaluator(options_config.evaluation_config)
    optimizer = SystemPromptGenerator(
        options_config, 
        evaluator, 
        retriever=retriever, 
        verbose=log_config.verbose
    )
    
    return optimizer.optimize()

def create_pipeline(trial_config: GenerationConfig, retriever: Any = None) -> Any:
    """Create generation pipeline from config"""
    logger = logging.getLogger("ragbuilder.generation.pipeline")
    logger.debug(f"Creating pipeline with config: {trial_config}")
    
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
        from langchain_core.output_parsers import StrOutputParser

        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        # Get initialized LLM directly from LLMConfig
        llm = trial_config.llm.llm
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