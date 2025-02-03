import logging
from dataclasses import dataclass
from pathlib import Path
import typing as t
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import json
import copy

from ragas.testset import TestsetGenerator, TestsetSample, Testset
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.persona import Persona
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.extractors import NERExtractor, SummaryExtractor, ThemesExtractor, EmbeddingExtractor
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

from openai import OpenAI, RateLimitError, APIError
from nemo_curator import OpenAIClient
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from ragas.testset.transforms.relationship_builders.traditional import OverlapScoreBuilder
from ragas.testset.transforms.relationship_builders.cosine import SummaryCosineSimilarityBuilder
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.single_hop.prompts import QueryAnswerGenerationPrompt as SingleHopPrompt
from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt as MultiHopPrompt


logger = logging.getLogger(__name__)

run_config = RunConfig(
    timeout=180,
    max_retries=10,
    max_wait=60,
    exception_types=(Exception, RateLimitError), 
    log_tenacity=True 
)

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    initial_testset_size: int = 10
    quality_threshold: float = 1.5  # Minimum average score to keep a Q&A pair
    semantic_similarity_threshold: float = 0.85  # Threshold for deduplication
    batch_size: int = 10
    cache_dir: Path = Path("cache")
    reward_model_name: str = "nvidia/nemotron-4-340b-reward"
    output_dir: Path = Path("output")

@dataclass
class EvaluatedSample:
    """Wrapper class to hold TestsetSample with its evaluation metrics"""
    sample: TestsetSample
    scores: dict
    avg_score: float

    @property
    def question(self) -> str:
        return self.sample.eval_sample.user_input
        
    @property
    def answer(self) -> str:
        return self.sample.eval_sample.reference

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation"""
        return {
            "question": self.question,
            "answer": self.answer,
            "synthesizer": self.sample.synthesizer_name,
            "avg_score": self.avg_score,
            **self.scores  # Unpack individual scores
        }

def parse_episode_info(filename: str) -> tuple[int, list[int]]:
    """Extract season and episode numbers from filename"""
    # Remove file extension
    filename = Path(filename).stem
    
    # Extract season number (first two digits)
    season = int(filename[:2])
    
    # Extract episode number(s)
    episode_part = filename[2:]
    episodes = []
    
    # Handle multi-episode files (e.g., "0212-0213")
    if '-' in episode_part:
        start, end = episode_part.split('-')
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(episode_part)]
        
    return season, episodes

@dataclass
class RLConfig:
    """Configuration for RL loop"""
    num_iterations: int = 3
    min_samples_per_iteration: int = 10
    exemplar_score_threshold: float = 1.6
    max_exemplars_per_iteration: int = 3
#     exemplar_template: str = """
# High-quality example Q&A pairs to learn from:

# {exemplars}

# Additional Instructions:
# 1. Learn from the style and depth of these examples
# 2. Focus on {focus_area} while maintaining similar quality
# 3. Ensure questions are diverse and non-repetitive
# 4. Maintain factual accuracy based on the show's content
# """
    exemplar_template: str = """
High-quality example Q&A pairs to learn from:

{exemplars}

Additional Instructions:
1. Learn from the style and depth of these examples
2. Ensure questions are diverse and non-repetitive
3. Maintain factual accuracy based on the show's content
"""

class RAGSyntheticDataGenerator:
    def __init__(
        self,
        llm: BaseRagasLLM,
        embedding_model: BaseRagasEmbeddings,
        reward_api_key: str,
        config: SyntheticDataConfig = None,
        rl_config: RLConfig = None
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.config = config or SyntheticDataConfig()
        self.rl_config = rl_config or RLConfig()

        self.ner_extractor = NERExtractor(
            llm=llm,
            max_num_entities=15
        )
        
        # Initialize reward model client
        self.reward_client = OpenAIClient(
            OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=reward_api_key,
            )
        )
        
        # Create directories
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.multi_hop_distribution = [
            (MultiHopAbstractQuerySynthesizer(llm=self.llm), 0.4),
            (MultiHopSpecificQuerySynthesizer(llm=self.llm), 0.4),
            (SingleHopSpecificQuerySynthesizer(llm=self.llm), 0.2)
        ]
        
        # Store original prompts
        self.single_hop_prompt = SingleHopPrompt()
        self.multi_hop_prompt = MultiHopPrompt()
        
    def load_transcripts(self, transcript_dir: Path) -> t.List[dict]:
        """Load transcripts from directory"""
        logger.info(f"Loading transcripts from {transcript_dir}")
        transcripts = []
        
        for file_path in tqdm(list(transcript_dir.glob("*.txt"))):
            # Parse season and episode info from filename
            season, episodes = parse_episode_info(file_path.name)
            
            # Read transcript
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create transcript entry
            transcripts.append({
                "content": content,
                "season": season,
                "episodes": episodes,
                "filename": file_path.name
            })
            
        logger.info(f"Loaded {len(transcripts)} transcripts")
        return transcripts
        
    async def build_knowledge_graph(self, transcripts: list[dict]) -> KnowledgeGraph:
        """Build knowledge graph with relationships for multi-hop queries"""
        logger.info("Building knowledge graph...")
        kg = KnowledgeGraph()
        
        # Create nodes with additional properties
        for transcript in tqdm(transcripts, desc="Creating nodes"):
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": transcript["content"],
                    "season": transcript["season"],
                    "episode": transcript["episodes"],
                    "filename": transcript["filename"]
                }
            )
            kg.nodes.append(node)
        
        # Define transforms 
        transforms=[
            # 1. Extract core properties
            Parallel(
                NERExtractor(llm=self.llm, max_num_entities=25),
                SummaryExtractor(llm=self.llm)
            ),
            
            # 2. Generate embeddings from summaries
            EmbeddingExtractor(
                embed_property_name="summary", 
                property_name="summary_embedding", 
                embedding_model=self.embedding_model
            ),
            
            # 3. Extract themes
            ThemesExtractor(llm=self.llm),
            
            # 4. Build relationships
            Parallel(
                OverlapScoreBuilder(
                    property_name="entities",
                    distance_threshold=0.75,
                    threshold=0.01,
                    filter_nodes=lambda node: bool(node.get_property("entities"))
                ),
                SummaryCosineSimilarityBuilder(
                    property_name="summary_embedding",
                    new_property_name="summary_similarity",
                    threshold=0.2,
                    filter_nodes=lambda node: bool(node.get_property("summary_embedding"))
                )
            ),
            
            # 5. Filter invalid nodes
            CustomNodeFilter(
                llm=self.llm,
                filter_nodes=lambda node: all([
                    node.get_property("summary"),
                    node.get_property("summary_embedding"),
                    node.get_property("entities")
                ])
            )
        ]

        try:
            await apply_transforms(kg, transforms, run_config=run_config)    
        except Exception as e:
            logger.error(f"Error applying transforms: {e}", exc_info=True)
            raise
        
        kg.save(f"friends_kg_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        logger.info(f"Created KG with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
        return kg
        
    def get_base_prompt(self, synthesizer_name: str) -> str:
        """Get base prompt based on synthesizer type"""
        if "multi_hop" in synthesizer_name.lower():
            return self.multi_hop_prompt.instruction
        return self.single_hop_prompt.instruction

    def format_exemplar(self, sample: EvaluatedSample) -> str:
        """Format a sample as an exemplar"""
#         return f"""
# Example {sample.sample.synthesizer_name}:
# Question: {sample.question}
# Answer: {sample.answer}
# Quality Metrics:
# - Helpfulness: {sample.scores.get('helpfulness', 'N/A')}
# - Correctness: {sample.scores.get('correctness', 'N/A')}
# - Coherence: {sample.scores.get('coherence', 'N/A')}
# """
        return f"""
Example {sample}:
Question: 
Answer: 
Quality Metrics:
- Helpfulness: 
- Correctness: 
- Coherence: 
"""

    def update_synthesizer_prompts(
        self,
        exemplars: list[EvaluatedSample],
        focus_area: str = None
    ) -> None:
        """Update prompts for all synthesizers with exemplars"""
        if not exemplars:
            return
            
        # Group exemplars by synthesizer type
        multi_hop_examples = [ex for ex in exemplars if "multi_hop" in ex.sample.synthesizer_name.lower()][:self.rl_config.max_exemplars_per_iteration]
        single_hop_examples = [ex for ex in exemplars if "single_hop" in ex.sample.synthesizer_name.lower()][:self.rl_config.max_exemplars_per_iteration]
        
        def create_exemplar_instruction(examples: list[EvaluatedSample]) -> str:
            """Create exemplar section of the instruction"""
            if not examples:
                return ""
            
            exemplar_text = "\n\n".join(self.format_exemplar(ex) for ex in examples)
            focus_section = f"\nFocus on: {focus_area}" if focus_area else ""
            
            return f"""
{self.rl_config.exemplar_template.format(exemplars=exemplar_text)}
{focus_section}
"""

        # Update prompts for each synthesizer
        for synthesizer, _ in self.multi_hop_distribution:
            is_multi_hop = "multi_hop" in synthesizer.__class__.__name__.lower()
            examples = multi_hop_examples if is_multi_hop else single_hop_examples
            
            if not examples:
                continue
            
            # Create a new PydanticPrompt instance with updated instruction
            base_instruction = self.get_base_prompt(synthesizer.__class__.__name__)
            exemplar_instruction = create_exemplar_instruction(examples)
            updated_instruction = f"{base_instruction}\n\n{exemplar_instruction}"
            
            # Create new prompt instance with updated instruction
            new_prompt = copy.deepcopy(synthesizer.generate_query_reference_prompt)
            new_prompt.instruction = updated_instruction
            
            # Update the synthesizer's prompt
            synthesizer.generate_query_reference_prompt = new_prompt

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def evaluate_quality(
        self, 
        question: str, 
        answer: str,
        synthesizer_name: str
    ) -> dict:
        """Evaluate Q&A pair quality using reward model"""
        messages = [
            {"role": "user", "content": self.get_base_prompt(synthesizer_name)},
            {"role": "assistant", "content": json.dumps({
                "query": question,
                "answer": answer
            }, indent=2)}
        ]
        
        try:
            rewards = self.reward_client.query_reward_model(
                messages=messages,
                model=self.config.reward_model_name
            )
            return rewards
        except Exception as e:
            logger.error(f"Error evaluating Q&A pair: {e}")
            raise
            
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def generate_initial_testset(self, knowledge_graph: KnowledgeGraph):
        """Generate testset with multi-hop queries"""
        logger.info("Generating testset with multi-hop queries...")
        
        # Define personas
        personas = [
            Persona(
                name="Friends Fan",
                role_description=(
                    "Enthusiastic viewer who knows the show well and asks detailed questions "
                    "about plot points, character relationships, running jokes, and specific "
                    "episodes. Interested in character development, memorable scenes, and "
                    "recurring themes throughout the series."
                )
            ),
            Persona(
                name="Casual Viewer",
                role_description=(
                    "Someone who has seen some episodes and asks basic questions about "
                    "main characters, major plot events, relationships between characters, "
                    "and popular episodes. Interested in understanding key moments and "
                    "character dynamics."
                )
            ),
            Persona(
                name="TV Critic",
                role_description=(
                    "Analyzes the show's themes, character development, narrative structure, "
                    "and cultural impact. Interested in storytelling techniques, social "
                    "commentary, character arcs, relationship dynamics, and how the show "
                    "reflects its era."
                )
            )
        ]
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embedding_model,
            knowledge_graph=knowledge_graph,
            persona_list=personas,
        )
        
        try:
            # Generate testset
            testset = generator.generate(
                testset_size=self.config.initial_testset_size,
                query_distribution=self.multi_hop_distribution
            )
            return testset
        except Exception as e:
            logger.error(f"Error generating testset: {e}")
            raise  # Let tenacity handle the retry
        
    def semantic_deduplication(self, samples: t.List[EvaluatedSample]) -> t.List[EvaluatedSample]:
        """Remove semantically similar questions using embeddings"""
        logger.info("Performing semantic deduplication...")
        
        if not samples:
            return []
            
        # Get embeddings for all questions
        questions = [sample.question for sample in samples]
        embeddings = self.embedding_model.embed_documents(questions)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Track which samples to keep
        to_keep = set(range(len(samples)))
        
        # Remove similar pairs
        for i in range(len(samples)):
            if i not in to_keep:
                continue
                
            for j in range(i + 1, len(samples)):
                if j not in to_keep:
                    continue
                    
                if similarity_matrix[i][j] > self.config.semantic_similarity_threshold:
                    # Keep the one with higher quality score
                    if samples[i].avg_score >= samples[j].avg_score:
                        to_keep.remove(j)
                    else:
                        to_keep.remove(i)
                        break
        
        deduped_samples = [samples[i] for i in sorted(to_keep)]
        logger.info(f"Reduced from {len(samples)} to {len(deduped_samples)} samples after deduplication")
        return deduped_samples
        
    def save_results(
        self, 
        accepted_samples: list[EvaluatedSample], 
        rejected_samples: list[EvaluatedSample]
    ) -> pd.DataFrame:
        """Save results to CSV files and return combined metrics DataFrame
        
        Returns:
            pd.DataFrame: Combined metrics for all samples
        """
        # Convert samples to DataFrames
        accepted_df = pd.DataFrame([sample.to_dict() for sample in accepted_samples])
        rejected_df = pd.DataFrame([sample.to_dict() for sample in rejected_samples])
        
        # Add acceptance status
        accepted_df["status"] = "accepted"
        rejected_df["status"] = "rejected"
        
        # Save individual files
        accepted_path = self.config.output_dir / "accepted_samples.csv"
        rejected_path = self.config.output_dir / "rejected_samples.csv"
        
        accepted_df.to_csv(accepted_path, index=False)
        rejected_df.to_csv(rejected_path, index=False)
        
        logger.info(f"Saved {len(accepted_samples)} accepted samples to {accepted_path}")
        logger.info(f"Saved {len(rejected_samples)} rejected samples to {rejected_path}")
        
        # Return combined metrics
        return pd.concat([accepted_df, rejected_df], ignore_index=True)
        
    def load_knowledge_graph(self, kg_path: Path) -> KnowledgeGraph:
        """Load knowledge graph from JSON file"""
        logger.info(f"Loading knowledge graph from {kg_path}")
        kg = KnowledgeGraph.load(kg_path)
        logger.info(f"Loaded knowledge graph with {len(kg.nodes)} nodes")
        return kg
    
    def get_focus_areas(self) -> list[str]:
        """Define focus areas for diverse question generation"""
        return [
            "character relationships and development",
            "memorable scenes and quotes",
            "running jokes and recurring themes",
            "plot developments and story arcs",
            "character backstories",
            "specific episodes or seasons",
            "locations and settings",
            "supporting characters",
        ]

    async def generate_with_rl(
        self, 
        transcript_dir: Path = None, 
        kg_path: Path = None
    ) -> tuple[Testset, pd.DataFrame]:
        """Main pipeline with RL loop for synthetic dataset generation"""
        exemplars: list[EvaluatedSample] = []
        all_samples: list[EvaluatedSample] = []
        focus_areas = self.get_focus_areas()
        
        # Get or build knowledge graph
        kg = (self.load_knowledge_graph(kg_path) if kg_path 
              else await self.build_knowledge_graph(self.load_transcripts(transcript_dir)))
        
        try:
            for iteration in range(self.rl_config.num_iterations):
                logger.info(f"Starting RL iteration {iteration + 1}/{self.rl_config.num_iterations}")
                
                # Update synthesizer prompts with exemplars
                # focus_area = focus_areas[iteration % len(focus_areas)]
                focus_area = None # For now, we don't want to focus on any specific area
                self.update_synthesizer_prompts(exemplars, focus_area)
                
                # Generate new samples
                testset = self.generate_initial_testset(kg)
                
                # Evaluate quality
                evaluated_samples: list[EvaluatedSample] = []
                for sample in tqdm(testset.samples, desc="Evaluating quality"):
                    try:
                        # Use correct base prompt for evaluation
                        # base_prompt = self.get_base_prompt(sample.synthesizer_name)
                        rewards = self.evaluate_quality(
                            question=sample.eval_sample.user_input,
                            answer=sample.eval_sample.reference,
                            synthesizer_name=sample.synthesizer_name
                        )
                        
                        if not rewards:
                            continue
                            
                        avg_score = sum(rewards.values()) / len(rewards)
                        evaluated_samples.append(EvaluatedSample(
                            sample=sample,
                            scores=rewards,
                            avg_score=avg_score
                        ))
                            
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}", exc_info=True)
                        continue
                
                # Update exemplars for next iteration
                exemplars = self.select_exemplars(evaluated_samples, exemplars)
                all_samples.extend(evaluated_samples)
                
                logger.info(
                    f"Iteration {iteration + 1} complete. "
                    f"Generated {len(evaluated_samples)} samples, "
                    f"Total exemplars: {len(exemplars)}"
                )
                
                if len(evaluated_samples) < self.rl_config.min_samples_per_iteration:
                    logger.warning(
                        f"Generated fewer than {self.rl_config.min_samples_per_iteration} "
                        "samples in this iteration. Consider adjusting parameters."
                    )
            
            # Final processing
            accepted = [es for es in all_samples 
                       if es.avg_score >= self.config.quality_threshold]
            rejected = [es for es in all_samples 
                       if es.avg_score < self.config.quality_threshold]
            
            final_samples = self.semantic_deduplication(accepted)
            metrics_df = self.save_results(final_samples, rejected)
            
            return Testset(samples=[es.sample for es in final_samples]), metrics_df
            
        except Exception as e:
            logger.error(f"Error in RL generation pipeline: {e}", exc_info=True)
            raise

    def select_exemplars(
        self, 
        new_samples: list[EvaluatedSample],
        existing_exemplars: list[EvaluatedSample]
    ) -> list[EvaluatedSample]:
        """Select diverse, high-quality samples as exemplars for next iteration
        
        Args:
            new_samples: New samples from current iteration
            existing_exemplars: Exemplars from previous iterations
        
        Returns:
            Combined list of selected exemplars
        """
        # Filter high-quality samples
        high_quality = [
            s for s in new_samples 
            if s.avg_score >= self.rl_config.exemplar_score_threshold
        ]
        
        if not high_quality:
            return existing_exemplars
        
        # Get embeddings for diversity comparison
        all_samples = high_quality + existing_exemplars
        questions = [s.question for s in all_samples]
        
        try:
            embeddings = self.embedding_model.embed_documents(questions)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Track selected samples
            selected_indices = set()
            num_existing = len(existing_exemplars)
            
            # First, keep existing exemplars
            selected_indices.update(range(num_existing))
            
            # Then select diverse new samples
            for i in range(num_existing, len(all_samples)):
                # Check similarity with already selected samples
                max_similarity = max(
                    similarity_matrix[i][j] 
                    for j in selected_indices
                ) if selected_indices else 0
                
                # Add if sufficiently different
                if max_similarity < 0.7:  # Diversity threshold
                    selected_indices.add(i)
                
                # Stop if we have enough exemplars
                if len(selected_indices) >= self.rl_config.max_exemplars_per_iteration:
                    break
            
            # Convert indices back to samples
            selected_samples = [all_samples[i] for i in sorted(selected_indices)]
            
            # If we have too many, prioritize by score
            if len(selected_samples) > self.rl_config.max_exemplars_per_iteration:
                selected_samples.sort(key=lambda x: x.avg_score, reverse=True)
                selected_samples = selected_samples[:self.rl_config.max_exemplars_per_iteration]
            
            logger.info(
                f"Selected {len(selected_samples)} exemplars "
                f"(kept {num_existing} existing, added {len(selected_samples) - num_existing} new)"
            )
            
            return selected_samples
            
        except Exception as e:
            logger.error(f"Error selecting exemplars: {e}", exc_info=True)
            # On error, keep existing exemplars and add top new samples up to limit
            high_quality.sort(key=lambda x: x.avg_score, reverse=True)
            combined = existing_exemplars + high_quality
            return combined[:self.rl_config.max_exemplars_per_iteration]

