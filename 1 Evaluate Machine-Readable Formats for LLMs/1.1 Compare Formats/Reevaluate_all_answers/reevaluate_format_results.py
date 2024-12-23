import os
import json
import logging
import time
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from evaluation_tracker import EvaluationTracker
from tenacity import retry, stop_after_attempt, wait_exponential

class EvaluationError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def evaluate_with_retry(evaluator, prompt):
    try:
        return evaluator.invoke(prompt).strip().lower()
    except Exception as e:
        logging.error(f"Connection error: {str(e)}")
        raise EvaluationError("Failed to connect to the model")

def initialize_ollama():
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            evaluator = OllamaLLM(model="gemma2:9b", temperature=0)
            # Test the connection
            evaluator.invoke("test")
            return evaluator
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Failed to initialize Ollama (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise RuntimeError("Failed to initialize Ollama after multiple attempts. Is the Ollama server running?")



# The model used to evaluate the responses had many misclassifications. 
# Since the results are stored in JSON files, we can re-evaluate the responses using a new model and save the updated results in new files. 
# This script re-evaluates the responses in all format directories 
# and saves the updated results.
def reevaluate_format_results(base_dir: str):
    """
    Re-evaluate results for all format directories.
    
    Args:
        base_dir (str): Base directory containing all analysis_context_robustness folders
        new_model_name (str): Name of the new model to use for evaluation
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        new_evaluator = initialize_ollama()
    except RuntimeError as e:
        logger.error(str(e))
        return
    tracker = EvaluationTracker()
    
    evaluation_template = PromptTemplate(
        input_variables=["question", "model_answer", "expected_answer"],
        template="""
        EVALUATION INSTRUCTIONS:
        - Compare the technical accuracy between Model's answer and Expected answer
        - Focus on core technical meaning
        - If the Model's answer is unsure or "Error", respond with 'no'
        - Respond only with 'yes' or 'no'
        
        Question: {question}
        Model's answer: {model_answer}
        Expected answer: {expected_answer}
        
        Answer (yes/no):
        """
    )
    
    # List of format directories to process
    format_dirs = [
        # "analysis_context_robustness_.csv",
        # "analysis_context_robustness_.json",
        # "analysis_context_robustness_.md",
        # "analysis_context_robustness_.pdf",
        # "analysis_context_robustness_.xml",
        "analysis_context_robustness_.yaml"
    ]
    
    for format_dir in format_dirs:
        dir_path = os.path.join(base_dir, format_dir)
        
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            continue
            
        logger.info(f"Processing directory: {format_dir}")
        
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        for file in tqdm(json_files, desc=f"Processing {format_dir}"):
            file_path = os.path.join(dir_path, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                modified = False
                for result in tqdm(results, desc=f"Evaluating {file}", leave=False):
                    if tracker.is_completed(file_path, result['question']):
                        continue
                        
                    try:
                        # Create a new evaluator instance for each question to prevent hallucinations
                        new_evaluator = initialize_ollama()
                        
                        eval_prompt = evaluation_template.format(
                            question=result['question'],
                            model_answer=result['response'],
                            expected_answer=result['expected_answer'][0]
                        )
                        
                        new_evaluation = evaluate_with_retry(new_evaluator, eval_prompt)
                        result['self_evaluation'] = new_evaluation
                        tracker.mark_completed(file_path, result['question'])
                        modified = True
                        
                    except EvaluationError:
                        logger.error(f"Skipping question due to evaluation error in {file}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error processing result in {file}: {str(e)}")
                        continue
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    logger.info(f"Updated results in: {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")

if __name__ == "__main__":
    base_directory = r"C:\llm\manual_reader\1 Evaluate Machine-Readable Formats for LLMs\1.1 Compare Formats"
    # new_model = "phi3:medium-128k"  # Specify your new model here
    
    reevaluate_format_results(base_directory)