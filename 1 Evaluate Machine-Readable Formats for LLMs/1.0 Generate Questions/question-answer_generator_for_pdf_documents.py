import os
import json
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import logging
import time
from typing import Dict, List

# question and answer generator for PDF documents using Ollama LLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_file_path(file_path: str):
    """
    Comprehensive file path debugging function
    
    :param file_path: Path to the file to check
    """
    print("\n--- File Path Debugging ---")
    print(f"Full Path: {file_path}")
    print(f"Absolute Path: {os.path.abspath(file_path)}")
    print(f"Normalized Path: {os.path.normpath(file_path)}")
    print(f"File Exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        # List contents of the directory
        dir_path = os.path.dirname(file_path)
        print(f"\nContents of directory {dir_path}:")
        try:
            for item in os.listdir(dir_path):
                print(f"- {item}")
        except Exception as e:
            print(f"Error listing directory contents: {e}")

class QuestionGenerator:
    def __init__(self, pdf_path: str, existing_questions_path: str):
        """
        Initialize the question generator with PDF path and existing questions file.
        
        :param pdf_path: Path to the PDF file to generate questions from
        :param existing_questions_path: Path to existing question-answer pairs JSON file
        """
        self.pdf_path = pdf_path
        self.existing_questions_path = existing_questions_path
        self.llm = OllamaLLM(model="llama3.1")
        
        # Template for question generation
        self.qa_template = PromptTemplate(
            input_variables=["page_content", "page_number"],
            template="""
            You are an expert at creating technical questions and answers from documentation.
            Below is the content from page {page_number} of a technical manual.
            Create ONE technical question with an ultra-brief answer from page {page_number}.

            Content:
            {page_content}

            Guidelines:
            - Answer must be concise and specific
            - Answer must be under 10 words
            - Focus on critical technical details
            - Include specific measurements or key facts
            - Provide precise information and use minimal language           
            
            Respond in this exact JSON format:
            {{"question": "your question here", "answer": "your concise answer here"}}
            """
        )
        
        # Load existing questions to avoid duplicates
        self.existing_questions = self.load_existing_questions()

    def load_existing_questions(self) -> set:
        """
        Load existing questions from the JSON file.
        
        :return: Set of existing question pages
        """
        try:
            # Ensure the file exists or create it
            if not os.path.exists(self.existing_questions_path):
                logger.warning(f"Creating new questions file: {self.existing_questions_path}")
                with open(self.existing_questions_path, 'w') as f:
                    json.dump([], f)
            
            with open(self.existing_questions_path, 'r') as f:
                existing_data = json.load(f)
                # If the JSON is in the new format with list of dictionaries
                if isinstance(existing_data, list):
                    return {item.get('page', 0) for item in existing_data}
                # If the JSON is in the old dictionary format
                return {page for _, (_, page) in existing_data.items()}
        except Exception as e:
            logger.error(f"Error loading existing questions: {str(e)}")
            return set()

    def load_pdf(self) -> List[dict]:
        """Load and process the PDF document."""
        try:
            # Comprehensive file path debugging
            debug_file_path(self.pdf_path)
            
            # Verify file exists before loading
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
            
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

    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate question-answer pairs for each page of the PDF."""
        pages = self.load_pdf()
        qa_pairs = []
        
        for page in pages:
            try:
                # Skip pages with very little content
                if len(page['content'].strip()) < 200:
                    continue
                
                # Skip pages already processed or outside desired range
                if (page['page'] < 23 or page['page'] > 141 or 
                    page['page'] in self.existing_questions):
                    continue
                
                logger.info(f"Processing page {page['page']}")
                
                # Generate question and answer for the page
                prompt = self.qa_template.format(
                    page_content=page['content'],
                    page_number=page['page'] + 1  # Convert to 1-based page numbering
                )
                
                # Get response from model
                response = self.llm.invoke(prompt)
                
                try:
                    # Parse JSON response
                    qa_data = json.loads(response)
                    
                    # Create dictionary in the requested format
                    qa_pair = {
                        "question": qa_data['question'],
                        "answer": qa_data['answer'],
                        "page": page['page'] + 1
                    }
                    
                    qa_pairs.append(qa_pair)
                    
                    # Add small delay to avoid overwhelming the model
                    time.sleep(0.5)
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response for page {page['page'] + 1}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing page {page['page'] + 1}: {str(e)}")
                continue
        
        return qa_pairs

    def save_qa_pairs(self, new_qa_pairs: List[Dict[str, str]], output_file: str = None):
        """
        Save the generated QA pairs to a file, merging with existing pairs.
        
        :param new_qa_pairs: List of new QA pairs to add
        :param output_file: Path to save the output file. If None, use the existing questions path.
        """
        output_file = output_file or self.existing_questions_path
        
        try:
            # Try to load existing data
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    # Ensure existing data is a list
                    if not isinstance(existing_data, list):
                        # If it's in the old dictionary format, convert to list
                        existing_data = [
                            {"question": q, "answer": v[0], "page": v[1]} 
                            for q, v in existing_data.items()
                        ]
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
        
        except Exception as e:
            logger.error(f"Error reading existing file: {str(e)}")
            existing_data = []
        
        # Merge existing and new data
        merged_data = existing_data + new_qa_pairs
        
        try:
            with open(output_file, 'w') as f:
                json.dump(merged_data, f, indent=4)
            logger.info(f"Successfully saved QA pairs to {output_file}")
        except Exception as e:
            logger.error(f"Error saving QA pairs: {str(e)}")

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths using os.path.normpath for consistent path handling
    pdf_path = os.path.normpath(os.path.join(script_dir, '..',  '..', 'manuals', 'EN-A148703540-2.pdf'))
    question_pairs_path = os.path.normpath(os.path.join(script_dir, '..',  '..', 'question_answer_pairs.json'))
    
    # Print paths for debugging
    print(f"Script Directory: {script_dir}")
    print(f"PDF Path: {pdf_path}")
    print(f"Questions Path: {question_pairs_path}")
    
    # Additional debugging information
    print("\n--- Path Existence Check ---")
    print(f"PDF Directory Exists: {os.path.exists(os.path.dirname(pdf_path))}")
    print(f"Manuals Directory Contents:")
    try:
        manuals_dir = os.path.dirname(pdf_path)
        for item in os.listdir(manuals_dir):
            print(f"- {item}")
    except Exception as e:
        print(f"Error listing manuals directory: {e}")
    
    generator = QuestionGenerator(pdf_path, question_pairs_path)
    
    try:
        # Generate QA pairs
        qa_pairs = generator.generate_qa_pairs()
        
        # Print results
        print("\nGenerated Question-Answer Pairs:")
        print("=" * 50)
        for qa_pair in qa_pairs:
            print(f"\nQuestion: {qa_pair['question']}")
            print(f"Answer: {qa_pair['answer']}")
            print(f"Page: {qa_pair['page']}")
        
        # Save results
        generator.save_qa_pairs(qa_pairs)
        # generator.save_qa_pairs(qa_pairs, output_file=new_question_pairs_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()