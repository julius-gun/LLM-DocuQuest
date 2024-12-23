import json
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import time
from typing import Dict, List, Tuple
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFAnalyzer:
    def __init__(self, pdf_path: str):
        """Initialize the PDF analyzer with the path to the PDF file."""
        self.pdf_path = pdf_path
        self.models = {
            "llama3.2:1b": OllamaLLM(model="llama3.2:1b"),
            "llama3.2": OllamaLLM(model="llama3.2"),
            "llama3.1": OllamaLLM(model="llama3.1"),
            "phi3:mini-128k": OllamaLLM(model="phi3:mini-128k"),
            "phi3:medium-128k": OllamaLLM(model="phi3:medium-128k")
            # "llama3.3:70b-instruct-q2_K": OllamaLLM(model="llama3.3:70b-instruct-q2_K")
        }

        # Create a separate evaluator model
        # self.evaluator_model = OllamaLLM(model="phi3:medium-128k")
        self.evaluator_model = OllamaLLM(model="llama3.1")

        # QUESTONS AND ANSWERS
        # file_path_questions = "/../question_answer_pairs.json"
        file_path_questions = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'question_answer_pairs.json')
        self.questions_and_answers = self.load_questions(file_path_questions)


        # Below is a Context from a technical manual. Please answer the question at the end based only on the context provided.
        # The context includes much unnecessary information. Therefore ignore tables. Include the relevant sentence in your answer, and only answer the question if you are very sure. 
        # Don't just copy the context but answer in a manner that makes sense. Double-check your answer before submitting. 
        # If possible answer in a few words. The answer should be concise and to the point. The answer should be not longer than 20 words.

        # Template for zero-shot learning
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        TASK: Extract a precise, concise answer from the given technical manual context.

        STRICT GUIDELINES:
        1. Read the entire context carefully
        2. Focus ONLY on the specific information related to the question
        3. Provide an extremely precise answer
        4. Match the expected answer format exactly
        5. If unsure, respond with "Unknown" or "Not found in context"

        CONTEXT PROCESSING RULES:
        - Ignore decorative elements, headers, and tables
        - Extract only factual, verifiable information
        - Prioritize numerical values and specific technical details
        - Do not add interpretation or extra explanation
        - Answer with less than 20 words

        Context:
        {context}

        Question: {question}

        Answer: [Carefully extract the EXACT information that directly answers the question, keeping it as brief and precise as possible]
            """
        )

        # Improved evaluation template for more accurate assessment
        self.evaluation_template = PromptTemplate(
            input_variables=["question", "model_answer", "expected_answer"],
            template="""
        ANSWER COMPARISON TASK:

        Evaluation Criteria:
        1. Compare the ESSENTIAL CORE TECHNICAL INFORMATION
        2. Allow SLIGHT VARIATIONS in:
        - Numerical values (within +-10% for measurements)
        - Unit representations (e.g., km/h vs kilometers per hour)
        - Phrasing and grammatical structure
        3. Ignore minor differences that do not impact technical meaning
        4. Focus on SUBSTANTIVE TECHNICAL ACCURACY

        Question: {question}
        Model's answer: {model_answer}
        Expected answer: {expected_answer}

        EVALUATION INSTRUCTIONS:
        - Respond 'yes' if the Model's answer and the Expected answer convey the SAME TECHNICAL MEANING
        - Consider 'yes' if differences are INSIGNIFICANT to the core technical content
        - Respond 'no' ONLY if there are MEANINGFUL differences that alter the technical understanding
        - Assess the SUBSTANCE of the information, not surface-level variations
        - Answer ONLY with yes or no
        - Don't provide additional information
        
        Answer (yes/no):
            """
        )
    def load_pdf(self) -> List[dict]:
        try:
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()

            processed_pages = []
            for page in pages:
                content = page.page_content
                page_num = page.metadata['page']
                processed_pages.append({
                    'content': content,
                    'page': page_num
                })
            return processed_pages
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def load_questions(self, file_path: str) -> dict:
        """Load questions and answers from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                questions_and_answers = {
                    item['question']: [item['answer'], item['page']]
                    for item in data
                }
                return questions_and_answers
        except Exception as e:
            logger.error(f"Error loading questions from file: {str(e)}")
            raise

    def get_context_around_page(self, pages: List[dict], target_page: int, context_pages: int) -> str:
        """
        Get content for pages around the target page.
        
        Args:
        - pages: List of page dictionaries
        - target_page: The page number of the target page
        - context_pages: Number of pages to include before and after the target page
        
        Returns:
        - Concatenated content of context pages
        """
        context_content = []
        for page in pages:
            # Calculate page range to include
            if (page['page'] >= target_page - context_pages and 
                page['page'] <= target_page + context_pages):
                context_content.append(f"[Page {page['page']}] {page['content']}")
        
        return "\n\n".join(context_content)

    def run_model_evaluation(self, context_pages_list: List[int]) -> Dict:
        """
        Run evaluation across models with different context page configurations.
        
        Args:
        - context_pages_list: List of number of pages to include around target page
        """
        all_results = {}
        pages = self.load_pdf()

        # Iterate through different context page configurations
        for context_pages in context_pages_list:
            logger.info(f"Testing with {context_pages} pages of context before and after target page")
            
            for model_name, model in self.models.items():
                logger.info(f"Evaluating model: {model_name}")
                model_results = []
                result_path = os.path.dirname(__file__) + "/analysis_context_robustness"
                os.makedirs(result_path, exist_ok=True)
                result_file = f"{result_path}/{model_name.replace(':', '_')}_results_{context_pages}_pages_context.json"
                
                # Load existing results if the file exists
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as file:
                        model_results = json.load(file)

                answered_questions = {result['question'] for result in model_results}

                for question, expected_answer in self.questions_and_answers.items():
                    if question in answered_questions:
                        logger.info(f"Skipping already answered question: {question}")
                        continue

                    try:
                        target_page = expected_answer[1] - 1  # Adjust for zero-indexing
                        page_content = self.get_context_around_page(pages, target_page, context_pages)

                        prompt = self.prompt_template.format(
                            context=page_content,
                            question=question
                        )

                        start_time = time.time()
                        response = model.invoke(prompt)
                        end_time = time.time()

                        eval_prompt = self.evaluation_template.format(
                            question=question,
                            model_answer=response,
                            expected_answer=expected_answer[0]
                        )

                        self_evaluation = self.evaluator_model.invoke(eval_prompt).strip().lower()

                        result = {
                            'context_pages': context_pages,
                            'question': question,
                            'expected_answer': expected_answer,
                            'response': response,
                            'time_taken': end_time - start_time,
                            'self_evaluation': self_evaluation
                        }

                        model_results.append(result)

                        # Save results to file after each answer
                        with open(result_file, 'w', encoding='utf-8') as file:
                            json.dump(model_results, file, indent=4, ensure_ascii=False)

                        # Print detailed information
                        print(f"Context Pages: {context_pages}")
                        print(f"Question: {question}")
                        print(f"Expected Answer: {result['expected_answer'][0]}\n")
                        print(f"Model Response: {response}")
                        print(f"Self-Evaluation (Correct?): {self_evaluation}\n")

                        # Calculate and print current accuracy
                        accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)
                        print(f"Current Accuracy: {accuracy:.1f}% ({correct_answers} out of {total_questions} questions answered correctly)")

                    except Exception as e:
                        logger.error(f"Error with model {model_name} on question: {question}")
                        logger.error(str(e))
                        model_results.append({
                            'context_pages': context_pages,
                            'question': question,
                            'expected_answer': expected_answer,
                            'response': f"Error: {str(e)}",
                            'time_taken': None,
                            'self_evaluation': 'error'
                        })

                all_results[(model_name, context_pages)] = model_results

        return all_results

    def calculate_accuracy(self, results: List[Dict]) -> Tuple[float, int, int]:
        """Calculate accuracy metrics from results."""
        total_questions = len(results)
        correct_answers = sum(1 for result in results if 'yes' in result['self_evaluation'].lower())
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        return accuracy, correct_answers, total_questions

    def format_results(self, results: Dict) -> str:
        """Format the results into a readable report."""
        report = "PDF Analysis Results - Context Robustness\n" + "="*50 + "\n\n"
        
        # Summarize results for each model and context configuration
        for (model_name, context_pages), model_results in results.items():
            accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)
            report += f"Model: {model_name}\n"
            report += f"Context Pages: {context_pages}\n"
            report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers)\n\n"
        
        # Detailed results section
        report += "\n" + "="*20 + "\n\n" + "Detailed Analysis\n" + "="*20 + "\n\n"
    
        for (model_name, context_pages), model_results in results.items():
            accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)
            
            report += f"Model: {model_name}\n"
            report += f"Context Pages: {context_pages}\n"
            report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers)\n"
            report += "-"*50 + "\n\n"

            for result in model_results:
                report += f"Question: {result['question']}\n"
                report += f"Expected Answer: {result['expected_answer'][0]}\n"
                report += f"Model Response: {result['response']}\n"
                report += f"Self-Evaluation (Correct?): {result['self_evaluation']}\n"
                if result['time_taken']:
                    report += f"Time taken: {result['time_taken']:.2f} seconds\n"
                report += "\n"

            report += "\n"

        return report

def main():
    # Initialize analyzer, first from parent folder, then from manuals folder
    # pdf_path = "/0 Getting Started with LangChain and PDF Analysis/manuals/EN-A148703540-2.pdf"
    pdf_path = os.path.dirname(__file__) + "/../manuals/EN-A148703540-2.pdf"
    analyzis_path = os.path.dirname(__file__) + "/analysis_context_robustness/analysis_results_context_robustness.txt"

    # Specify different context page configurations to test
    context_pages_list = [0, 1, 3, 5]

    analyzer = PDFAnalyzer(pdf_path)
    try:
        # Run evaluation with different context page configurations
        results = analyzer.run_model_evaluation(context_pages_list)

        # Format and display results
        report = analyzer.format_results(results)
        print(report)

        # Save results to file
        os.makedirs(os.path.dirname(analyzis_path), exist_ok=True)
        with open(analyzis_path, "w", encoding='utf-8') as f:
            f.write(report)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()