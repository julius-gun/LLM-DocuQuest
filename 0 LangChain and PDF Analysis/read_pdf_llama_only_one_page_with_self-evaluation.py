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
            "llama3.2:1b": OllamaLLM(model="llama3.2:1b")
            # "llama3.2": OllamaLLM(model="llama3.2"),
            # "llama3.1": OllamaLLM(model="llama3.1"),
            # "phi3:mini-128k": OllamaLLM(model="phi3:mini-128k"),
            # "phi3:medium-128k": OllamaLLM(model="phi3:medium-128k"),
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

    def get_page_content(self, pages: List[dict], target_page: int) -> str:
        """Get content for a specific page number."""
        for page in pages:
            if page['page'] == target_page:
                return f"[Page {page['page']}] {page['content']}"
        return ""

    def calculate_accuracy(self, results: List[Dict]) -> Tuple[float, int, int]:
        """Calculate accuracy metrics from results."""
        total_questions = len(results)
        correct_answers = sum(1 for result in results if 'yes' in result['self_evaluation'].lower())
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        return accuracy, correct_answers, total_questions

    def run_model_evaluation(self) -> Dict:
        """Run evaluation across all models for given questions."""
        results = {}
        pages = self.load_pdf()

        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = []
            result_path = os.path.dirname(__file__) + "/analysis_one_page"
            result_file = result_path + f"/{model_name.replace(':', '_')}_results.json"
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
                    target_page = expected_answer[1] - 1
                    page_content = self.get_page_content(pages, target_page)

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

                    # Print question and answer
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
                        'question': question,
                        'expected_answer': expected_answer,
                        'response': f"Error: {str(e)}",
                        'time_taken': None,
                        'self_evaluation': 'error'
                    })

            results[model_name] = model_results

        return results

    def format_results(self, results: Dict) -> str:
        """Format the results into a readable report."""
        report = "PDF Analysis Results\n" + "="*20 + "\n\n"
        
        # Summary section
        for model_name, model_results in results.items():
            # Calculate accuracy metrics
            accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)
            report += f"Model: {model_name}\n" 
            report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers). " + "-"*50 + "\n\n"
        
        report += "\n" + "="*20 + "\n\n" + "Analysis for each model\n" + "="*20 + "\n\n"
    
        # Detailed results section
        for model_name, model_results in results.items():
            # Calculate accuracy metrics
            accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)
            report += f"Model: {model_name}:\n" 
            report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers). "+ "-"*50 + "\n\n"

            report += "-"*20 + "\n\n"
            report += f"Detailed Results\n"
            report += "-"*20 + "\n\n"
            report += f"Model: {model_name}\n" + "-"*50 + "\n"

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
    analyzis_path = os.path.dirname(__file__) + "/analysis_one_page/analysis_results_one_page_input_with_self_evaluation.txt"

    analyzer = PDFAnalyzer(pdf_path)
    try:
        # Run evaluation
        results = analyzer.run_model_evaluation()

        # Format and display results
        report = analyzer.format_results(results)
        print(report)

        # Optionally save results to file
        with open(analyzis_path, "w", encoding='utf-8') as f:
            f.write(report)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()