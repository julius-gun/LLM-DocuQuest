# Goal 1: Evaluate Machine-Readable Formats
## Multi-Format Document Analyzer for LLMs

This code evaluates the performance of various Large Language Models (LLMs) in answering questions based on a user manual provided in different file formats (PDF, CSV, JSON, Markdown, XML, YAML). It analyzes how well LLMs can extract information and answer questions accurately when given the manual in these different formats.

The manual being used in this example can be downloaded [here](https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf).

## Key Features:
*   **[PDF Question and Answer Generator using Ollama LLMs](/1%20Evaluate%20Machine-Readable%20Formats%20for%20LLMs/1.0%20Generate%20Questions/question-answer_generator_for_pdf_documents.py)**
    *   This code automatically generates question-answer pairs from PDF documents, specifically technical manuals, using Ollama Large Language Models (LLMs). It's designed to extract key technical information and create concise, relevant questions and answers.
*   **[PDF Question and Answer Generator using Google Gemini API](/1%20Evaluate%20Machine-Readable%20Formats%20for%20LLMs/1.0%20Generate%20Questions/question-answer_generator_for_pdf_documents_Gemini.py)**
    *   This code automatically generates question-answer pairs from PDF documents while using the Google Gemini API
*   **[Format Conversion and Loading:](/1%20Evaluate%20Machine-Readable%20Formats%20for%20LLMs/1.1%20Compare%20Formats/PDF_conversion.ipynb)**
    *   Loads a user manual from a PDF file.
    *   Converts the manual into CSV, JSON, Markdown, XML, and YAML formats.
*   **[LLM Document Analyzer:](/1%20Evaluate%20Machine-Readable%20Formats%20for%20LLMs/1.1%20Compare%20Formats/multi-format_document_analyzer.py)**
    *   Utilizes the `langchain_ollama` library to interact with Ollama models.
    *   Uses a set of pre-defined [questions and expected answers related to the manual](/1%20Evaluate%20Machine-Readable%20Formats%20for%20LLMs/question_answer_pairs.json).
    *   Prompts LLMs with questions about the manual, providing varying amounts of context (e.g., 0, 1 or 3 pages around the answer).
    *   Supports multiple LLMs, including different versions of Llama3 and Phi3.
    *   **Evaluation:**
        *   Uses a separate "evaluator" LLM (Gemma2) to assess the accuracy of the answers provided by the main LLMs.
        *   Calculates accuracy based on whether the evaluator deems the answer correct ("yes") or not ("no").
        *   Measures the time taken for each model to answer each question.
    *   **Results and Visualization:**
        *   Saves detailed results for each model, context length, and question in JSON files.
        *   Generates a summary report comparing the accuracy of different models and formats.
        *   Creates visualizations (plots) showing how accuracy changes with different context lengths for each model and format.
        *   Exports the visualization data to a CSV file for further analysis.
 #### Models tested include
 | Model             | Number of Parameters | Maximal Context Length |
|-------------------|----------------------|----------------|
| llama3.2:1b       | 1b                   | 128k           |
| llama3.2:3b       | 3b                   | 128k           |
| llama3.1          | 8b                   | 128k           |
| phi3:mini-128k    | 3.8b                 | 128k           |
| phi3:medium-128k  | 14b                  | 128k           |
| gemma2:9b (used only for evaluation)         | 9b                   | 8k           |

<p align="left">
  <img src="1 Evaluate Machine-Readable Formats for LLMs/1.1 Compare Formats/accuracy_analysis/accuracy_plot_phi3_mini-128k.png" alt="Accuracy Comparison for phi3 mini-128k" width="500"/>
  <br>
  <img src="1 Evaluate Machine-Readable Formats for LLMs/1.1 Compare Formats/accuracy_analysis/accuracy_plot_phi3_medium-128k.png" alt="Accuracy Comparison for phi3 medium-128k" width="500"/>
  <br>
  <img src="1 Evaluate Machine-Readable Formats for LLMs/1.1 Compare Formats/accuracy_analysis/accuracy_plot_llama3.2_1b.png" alt="Accuracy Comparison for llama3.2 1b" width="500"/>
  <br>
  <img src="1 Evaluate Machine-Readable Formats for LLMs/1.1 Compare Formats/accuracy_analysis/accuracy_plot_llama3.2.png" alt="Accuracy Comparison for llama3.2" width="500"/>
  <br>
  <img src="1 Evaluate Machine-Readable Formats for LLMs/1.1 Compare Formats/accuracy_analysis/accuracy_plot_llama3.1.png" alt="Accuracy Comparison for llama3.1" width="500"/>
  <br>
</p>
