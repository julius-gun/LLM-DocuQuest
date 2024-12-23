import os
import json
import csv
import logging
import numpy as np

# import pandas as pd
import yaml
import xml.etree.ElementTree as ET
import markdown
from typing import Dict, List, Tuple
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import time
import traceback
import matplotlib.pyplot as plt


def generate_visualization_data(consolidated_results):
    """
    Process consolidated results into a format suitable for visualization

    Parameters:
    - consolidated_results (dict): Results from multi-format document analysis

    Returns:
    - dict: Processed data for visualization and CSV export
    """
    visualization_data = []

    for file_format, model_results in consolidated_results.items():
        for (model_name, context_pages), results in model_results.items():
            # Calculate accuracy
            total_questions = len(results)
            correct_answers = sum(
                1
                for result in results
                if "yes" in result.get("self_evaluation", "").lower()
            )
            accuracy = (
                (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            )

            visualization_data.append(
                {
                    "file_format": os.path.splitext(file_format)[1][
                        1:
                    ],  # Remove the dot from extension
                    "model": model_name,
                    "context_pages": context_pages,
                    "total_questions": total_questions,
                    "correct_answers": correct_answers,
                    "accuracy": round(accuracy, 2),
                }
            )

    return visualization_data


def save_visualization_data(visualization_data, output_dir):
    """
    Save visualization data to CSV and generate plots for each model

    Parameters:
    - visualization_data (list): Processed data for visualization
    - output_dir (str): Directory to save outputs
    """
    # CSV Export
    csv_path = os.path.join(output_dir, "accuracy_analysis.csv")
    fieldnames = [
        "file_format",
        "model",
        "context_pages",
        "total_questions",
        "correct_answers",
        "accuracy",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visualization_data)

    # Prepare data for plotting
    models = sorted(set(data["model"] for data in visualization_data))
    file_formats = sorted(set(data["file_format"] for data in visualization_data))
    context_pages = sorted(set(data["context_pages"] for data in visualization_data))

    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(file_formats)))

    #     for row in visualization_data:
    #         writer.writerow(row)

    # # Define color and line style mappings
    # context_page_colors = {0: "red", 1: "blue", 3: "green", 5: "purple"}

    # line_styles = {
    #     "csv": "-",
    #     "json": "--",
    #     "md": ":",
    #     "xml": "-.",
    #     "yaml": "-",
    #     "yml": "-",
    #     "pdf": "-",
    # }

    # # Group data by model
    # models = set(data["model"] for data in visualization_data)
    # Create a plot for each model

    for model in models:
        plt.figure(figsize=(12, 8))

        # Filter data for current model
        model_data = [d for d in visualization_data if d["model"] == model]

        # Plot lines for each file format
        for idx, format_name in enumerate(file_formats):
            format_data = [d for d in model_data if d["file_format"] == format_name]
            if format_data:
                # Sort by context pages to ensure correct line plotting
                format_data.sort(key=lambda x: x["context_pages"])

                x = [d["context_pages"] for d in format_data]
                y = [d["accuracy"] for d in format_data]

                plt.plot(
                    x,
                    y,
                    "o-",
                    label=format_name,
                    color=colors[idx],
                    linewidth=2,
                    markersize=8,
                )

        plt.title(f"Accuracy by Context Pages for {model}", pad=20, size=14)
        plt.xlabel("Number of Context Pages", size=12)
        plt.ylabel("Accuracy (%)", size=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="File Format", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Set x-axis ticks to show only the actual context page values
        plt.xticks(context_pages)
        plt.ylim(0, 100)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            output_dir, f'accuracy_plot_{model.replace(":", "_")}.png'
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()


class MultiFormatDocumentAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the multi-format document analyzer.

        Args:
            file_path (str): Path to the document file
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()

        # Configure logging
        # logging.basicConfig(level=logging.INFO)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Set up LLM model configurations
        self.temperature = 0
        self.context_window = 131072
        
        # Remove model initialization from __init__
        self.model_configs = {
            "llama3.2:1b": {"name": "llama3.2:1b"},
            "llama3.2": {"name": "llama3.2"},
            "llama3.1": {"name": "llama3.1"},
            "phi3:mini-128k": {"name": "phi3:mini-128k"},
            "phi3:medium-128k": {"name": "phi3:medium-128k"}
        }
        self.evaluater_model = "gemma2:9b"
        # Configure LLM models
        # self.models = {
        #     "llama3.2:1b": OllamaLLM(model="llama3.2:1b", temperature=temperature),
        #     "llama3.2": OllamaLLM(model="llama3.2", temperature=temperature),
        #     "llama3.1": OllamaLLM(model="llama3.1", temperature=temperature),
        #     "phi3:mini-128k": OllamaLLM(
        #         model="phi3:mini-128k", temperature=temperature
        #     ),
        #     "phi3:medium-128k": OllamaLLM(
        #         model="phi3:medium-128k", temperature=temperature
        #     ),
        #     "gemma2:9b": OllamaLLM(model="gemma2", temperature=temperature),
        #     # "llama3.3:70b-instruct-q2_K": OllamaLLM(model="llama3.3:70b-instruct-q2_K")
        # }

        # Create a separate evaluator model
        # self.evaluator_model = OllamaLLM(model="llama3.2:1b")
        # self.evaluator_model = OllamaLLM(model="phi3:medium-128k")

        # Set up prompt templates
        self._setup_prompt_templates()

        # Load questions and answers
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # questions_file = os.path.join(current_dir, '..', 'question_answer_pairs.json')
        questions_file = r"C:\llm\manual_reader\question_answer_pairs.json"
        self.questions_and_answers = self._load_questions(questions_file)
        # print("self.questions_and_answers: ", self.questions_and_answers)
    def _create_fresh_model(self, model_name: str) -> OllamaLLM:
        """Create a new instance of the model with fresh context."""
        return OllamaLLM(model=self.model_configs[model_name]["name"], temperature=self.temperature, num_ctx=self.context_window)

    def _create_fresh_evaluator(self) -> OllamaLLM:
        """Create a new instance of the evaluator model with fresh context."""
        return OllamaLLM(model=self.evaluater_model, temperature=self.temperature)

    def _setup_prompt_templates(self):
        """Setup prompt templates for analysis and evaluation."""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        TASK: Extract a precise, concise answer from the given context.

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
            """,
        )

        # Improved evaluation template for more accurate assessment
        self.evaluation_template = PromptTemplate(
            input_variables=["question", "model_answer", "expected_answer"],
            template="""
        ANSWER COMPARISON TASK:

        Evaluation Criteria:
        1. Compare the ESSENTIAL CORE TECHNICAL INFORMATION
        2. Allow SLIGHT VARIATIONS in:
        - Numerical values (within +-5% for measurements)
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
            """,
        )

    def load_document(self) -> List[Dict[str, str]]:
        """
        Load document based on file extension.

        Returns:
            List of dictionaries with page content
        """
        try:
            if self.file_extension == ".csv":
                return self._load_csv()
            elif self.file_extension == ".json":
                return self._load_json()
            elif self.file_extension == ".md":
                return self._load_markdown()
            elif self.file_extension == ".xml":
                return self._load_xml()
            elif self.file_extension == ".yaml" or self.file_extension == ".yml":
                return self._load_yaml()
            elif self.file_extension == ".pdf" or self.file_extension == ".pdf":
                return self._load_pdf()
            else:
                raise ValueError(f"Unsupported file format: {self.file_extension}")
        except Exception as e:
            self.logger.error(f"Error loading document: {str(e)}")
            raise

    def _load_pdf(self) -> List[Dict[str, str]]:
        """
        Load PDF file and convert to the required format.
        Returns a list of dictionaries containing page content.
        """
        try:
            loader = PyPDFLoader(
                self.file_path
            )  # Changed from self.pdf_path to self.file_path
            pages = loader.load()

            processed_pages = []
            for page in pages:
                content = page.page_content
                page_num = page.metadata["page"]
                processed_pages.append(
                    {
                        "page": f"Page {page_num + 1}",  # Add 1 since PDF pages typically start at 1, not 0
                        "content": content,
                    }
                )
            return processed_pages
        except Exception as e:
            self.logger.error(f"Error loading PDF: {str(e)}")
            raise

    def _load_csv(self) -> List[Dict[str, str]]:
        """Load CSV file with robust parsing for multi-line content"""
        try:
            # Read the entire file content
            with open(self.file_path, "r", encoding="utf-8") as f:
                csv_content = f.read()

            # Split the content into lines
            lines = csv_content.split("\n")

            # Process lines into pages
            pages = []
            current_page = None
            current_content = []

            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Check if this line starts with a new page
                if line.startswith("Page "):
                    # If we had a previous page, save it
                    if current_page is not None:
                        pages.append(
                            {
                                "page": current_page,
                                "content": " ".join(current_content).strip(),
                            }
                        )

                    # Start a new page
                    current_page = line.split(",")[0]
                    current_content = [line.split(",", 1)[1]]
                else:
                    # Continue collecting content for the current page
                    current_content.append(line)

            # Add the last page
            if current_page is not None:
                pages.append(
                    {"page": current_page, "content": " ".join(current_content).strip()}
                )

            return pages

        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise

    def _load_json(self) -> List[Dict[str, str]]:
        """Load JSON file"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [{"page": page, "content": content} for page, content in data.items()]

    def _load_markdown(self) -> List[Dict[str, str]]:
        """Load Markdown file"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        pages = markdown.markdown(text).split("<h2>")
        return [
            {"page": page.split("</h2>")[0], "content": page.split("</h2>")[1].strip()}
            for page in pages[1:]
            if page.strip()
        ]

    def _load_xml(self) -> List[Dict[str, str]]:
        """Load XML file"""
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        return [
            {"page": page.get("name"), "content": page.find("Content").text}
            for page in root.findall("Page")
        ]

    def _load_yaml(self) -> List[Dict[str, str]]:
        """Load YAML file"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [{"page": page, "content": content} for page, content in data.items()]

    def _load_questions(self, file_path: str) -> dict:
        """Load questions and answers from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return {
                    item["question"]: [item["answer"], item["page"]] for item in data
                }
        except Exception as e:
            self.logger.error(f"Error loading questions from file: {str(e)}")
            raise

    def get_context_around_page(
        self, pages: List[Dict[str, str]], target_page: int, context_pages: int
    ) -> str:
        """
        Get content for pages around the target page or full document.
        
        Args:
        - pages: List of page dictionaries
        - target_page: The page number of the target page
        - context_pages: Number of pages to include before and after target page, or 'full document' for full document
        
        Returns:
        - Concatenated content of context pages
        """
        if context_pages == 'full document':
            # Return full document content
            return "\n\n".join(f"[{page['page']}] {page['content']}" for page in pages)
        context_content = []
        for page in pages:
            try:
                page_num = int(page["page"].replace("Page ", ""))
                if (
                    page_num >= target_page - context_pages
                    and page_num <= target_page + context_pages
                ):
                    context_content.append(f"[{page['page']}] {page['content']}")
            except ValueError:
                # Fallback for pages that might not have a numeric identifier
                context_content.append(f"[{page['page']}] {page['content']}")

        return "\n\n".join(context_content)

    def run_model_evaluation(self, context_pages_list: List[int]) -> Dict:
        """
        Run evaluation across models with different context page configurations.

        Args:
        - context_pages_list: List of number of pages to include around target page
            Use 'full document' in context_pages_list to indicate full document context.

        """
        all_results = {}
        pages = self.load_document()

        # keep track of total tasks and completed tasks
        total_tasks = len(context_pages_list) * len(self.model_configs) * len(self.questions_and_answers)
        completed_tasks = 0

        # print(pages) # Debugging //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Iterate through different context page configurations
        for context_pages in context_pages_list:
            context_desc = "full document" if context_pages == 'full document' else f"{context_pages} pages"

            self.logger.info(
                f"Testing with {context_desc} of context"
            )

            for model_name in self.model_configs:
                self.logger.info(f"Evaluating model: {model_name}")
                model_results = []
                result_path = (
                    os.path.dirname(__file__)
                    + f"/analysis_context_robustness_{self.file_extension}"
                )
                os.makedirs(result_path, exist_ok=True)
                context_identifier = "full" if context_pages == -1 else str(context_pages)
                result_file = f"{result_path}/{model_name.replace(':', '_')}_results_{context_identifier}_pages_context.json"

                # Load existing results if the file exists
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as file:
                        model_results = json.load(file)

                answered_questions = {result["question"] for result in model_results}
                # Update completed tasks with already answered questions
                completed_tasks += len(answered_questions)

                for question, expected_answer in self.questions_and_answers.items():
                    if question in answered_questions:
                        self.logger.info(
                            f"Skipping already answered question: {question}"
                        )
                        continue

                    try:
                        # Create fresh model instances for each question
                        current_model = self._create_fresh_model(model_name)
                        current_evaluator = self._create_fresh_evaluator()

                        target_page = expected_answer[1]  # Actual page number
                        # print("target_page: ", target_page)
                        page_content = self.get_context_around_page(
                            pages, target_page, context_pages
                        )
                        # # append the page_content to a separate txt file, stating the current extension and the context pages
                        # with open(
                        #     f"{result_path}/{self.file_extension}_{context_pages}_pages_context.txt",
                        #     "a",
                        #     encoding="utf-8",
                        # ) as file:
                        #     file.write(page_content + "\n\n")
                        prompt = self.prompt_template.format(
                            context=page_content, question=question
                        )

                        start_time = time.time()
                        response = current_model.invoke(prompt)
                        end_time = time.time()

                        eval_prompt = self.evaluation_template.format(
                            question=question,
                            model_answer=response,
                            expected_answer=expected_answer[0],
                        )

                        self_evaluation = (
                            current_evaluator.invoke(eval_prompt).strip().lower()
                        )

                        result = {
                            "context_pages": context_pages,
                            "question": question,
                            "expected_answer": expected_answer,
                            "response": response,
                            "time_taken": end_time - start_time,
                            "self_evaluation": self_evaluation,
                        }

                        model_results.append(result)

                        # Save results to file after each answer
                        with open(result_file, "w", encoding="utf-8") as file:
                            json.dump(model_results, file, indent=4, ensure_ascii=False)

                        # Print detailed information
                        # print(f"Context Pages: {context_pages}")
                        # print(f"Question: {question}")
                        # print(f"Expected Answer: {result['expected_answer'][0]}\n")
                        # print(f"Model Response: {response}")
                        # print(f"Self-Evaluation (Correct?): {self_evaluation}\n")

                        # # Calculate and print current accuracy
                        accuracy, correct_answers, total_questions = (
                            self.calculate_accuracy(model_results)
                        )
                        print(
                            f"Current Accuracy: {accuracy:.1f}% ({correct_answers} out of {total_questions} questions answered correctly)"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error with model {model_name} on question: {question}"
                        )
                        self.logger.error(str(e))
                        model_results.append(
                            {
                                "context_pages": context_pages,
                                "question": question,
                                "expected_answer": expected_answer,
                                "response": f"Error: {str(e)}",
                                "time_taken": None,
                                "self_evaluation": "error",
                            }
                        )
                    # Update completed tasks and print progress
                    completed_tasks += 1
                    progress_percentage = (completed_tasks / total_tasks) * 100
                    print(f"Progress: {progress_percentage:.2f}%")
                    # Explicitly delete model instances to ensure clean context
                    del current_model
                    del current_evaluator

                # Calculate and print current accuracy after all questions
                accuracy, correct_answers, total_questions = self.calculate_accuracy(
                    model_results
                )
                print(
                    f"{self.file_extension}: Final Accuracy for model {model_name} with {context_pages} context pages: {accuracy:.1f}% ({correct_answers} out of {total_questions} questions answered correctly)"
                )
                all_results[(model_name, context_pages)] = model_results

        return all_results

    def calculate_accuracy(self, results: List[Dict]) -> Tuple[float, int, int]:
        """Calculate accuracy metrics from results."""
        total_questions = len(results)
        correct_answers = sum(
            1 for result in results if "yes" in result["self_evaluation"].lower()
        )
        accuracy = (
            (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        )
        return accuracy, correct_answers, total_questions

    def format_results(self, results: Dict) -> str:
        """Format the results into a readable report."""
        report = (
            f"Multiformat Document Analysis Results - {self.file_extension}\n"
            + "=" * 50
            + "\n\n"
        )

        # Summarize results for each model and context configuration
        for (model_name, context_pages), model_results in results.items():
            accuracy, correct_answers, total_questions = self.calculate_accuracy(
                model_results
            )
            report += f"Model: {model_name}\n"
            report += f"Context Pages: {context_pages}\n"
            report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers)\n\n"

        # # Detailed results section
        # report += "\n" + "="*20 + "\n\n" + "Detailed Analysis\n" + "="*20 + "\n\n"

        # for (model_name, context_pages), model_results in results.items():
        #     accuracy, correct_answers, total_questions = self.calculate_accuracy(model_results)

        #     report += f"Model: {model_name}\n"
        #     report += f"Context Pages: {context_pages}\n"
        #     report += f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions} correct answers)\n"
        #     report += "-"*50 + "\n\n"

        #     for result in model_results:
        #         report += f"Question: {result['question']}\n"
        #         report += f"Expected Answer: {result['expected_answer'][0]}\n"
        #         report += f"Model Response: {result['response']}\n"
        #         report += f"Self-Evaluation (Correct?): {result['self_evaluation']}\n"
        #         if result['time_taken']:
        #             report += f"Time taken: {result['time_taken']:.2f} seconds\n"
        #         report += "\n"

        #     report += "\n"

        return report


def main():
    # List of file paths for different formats
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # file_formats = [
    #     # os.path.dirname(__file__) + "/../manuals/EN-A148703540-2.pdf",
    #     # os.path.dirname(__file__) + "/../converted_manuals/CSV/EN-A148703540-2.csv",
    #     # os.path.dirname(__file__) + "/../converted_manuals/JSON/EN-A148703540-2.json",
    #     # os.path.dirname(__file__) + "/../converted_manuals/Markdown/EN-A148703540-2.md",
    #     # os.path.dirname(__file__) + "/../converted_manuals/XML/EN-A148703540-2.xml",
    #     # os.path.dirname(__file__) + "/../converted_manuals/YAML/EN-A148703540-2.yaml"
    #     os.path.join(base_dir, '..', 'converted_manuals', 'CSV', 'EN-A148703540-2.csv'),
    #     os.path.join(base_dir, '..', 'converted_manuals', 'JSON', 'EN-A148703540-2.json'),
    #     os.path.join(base_dir, '..', 'converted_manuals', 'Markdown', 'EN-A148703540-2.md'),
    #     os.path.join(base_dir, '..', 'converted_manuals', 'XML', 'EN-A148703540-2.xml'),
    #     os.path.join(base_dir, '..', 'converted_manuals', 'YAML', 'EN-A148703540-2.yaml')
    # ]
    file_formats = [
        r"C:\llm\manual_reader\manuals\EN-A148703540-2.pdf",
        r"C:\llm\manual_reader\converted_manuals\CSV\EN-A148703540-2.csv",
        r"C:\llm\manual_reader\converted_manuals\JSON\EN-A148703540-2.json",
        r"C:\llm\manual_reader\converted_manuals\Markdown\EN-A148703540-2.md",
        r"C:\llm\manual_reader\converted_manuals\XML\EN-A148703540-2.xml",
        r"C:\llm\manual_reader\converted_manuals\YAML\EN-A148703540-2.yaml",
    ]

    # Add debug logging to verify paths

    logging.basicConfig(level=logging.INFO)
    # no logging
    # logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(__name__)

    # Print out paths for verification
    for path in file_formats:
        logger.debug(f"Checking path: {path}")
        logger.debug(f"Path exists: {os.path.exists(path)}")

    # Specify different context page configurations to test
    context_pages_list = [0, 1, 3, 5, 'full document']

    # Consolidated results
    consolidated_results = {}

    for file_path in file_formats:
        try:
            # Verify file exists before processing
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue

            logger.info(f"Processing file: {file_path}")

            analyzer = MultiFormatDocumentAnalyzer(file_path)
            results = analyzer.run_model_evaluation(context_pages_list)

            # Format and save results for each file format
            # report = analyzer.format_results(results)
            # print(report) # Debugging //////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # Store results for further analysis
            consolidated_results[os.path.basename(file_path)] = results

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            traceback.print_exc()

    # # Write a summary of the results
    summary_report = "\n\n".join(
        [analyzer.format_results(results) for results in consolidated_results.values()]
    )
    # create a file
    with open(
        os.path.join(base_dir, "summary_report.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(summary_report)
    # print(summary_report)

    # Save visualization data
    output_dir = os.path.join(os.path.dirname(__file__), "accuracy_analysis")
    os.makedirs(output_dir, exist_ok=True)

    visualization_data = generate_visualization_data(consolidated_results)
    save_visualization_data(visualization_data, output_dir)

    print(f"Visualization data saved to {output_dir}")

    return consolidated_results
    # reevaluate_format_results(base_directory, new_model)


if __name__ == "__main__":
    main()
