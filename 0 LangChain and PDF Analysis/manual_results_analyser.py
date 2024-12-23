import os
import json
import hashlib

class ManualEvaluationTracker:
    def __init__(self, input_directory, output_directory):
        """
        Initialize the manual evaluation tracking system.
        
        Args:
            input_directory (str): Directory containing input JSON files
            output_directory (str): Directory to save processed results
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Path for tracking processed files and questions
        self.progress_file_path = os.path.join(output_directory, 'evaluation_progress.json')
        
        # Load or initialize progress tracking
        self.progress = self._load_progress()
    
    def _load_progress(self):
        """
        Load existing progress or initialize a new progress tracking dictionary.
        
        Returns:
            dict: Progress tracking information
        """
        if os.path.exists(self.progress_file_path):
            with open(self.progress_file_path, 'r') as f:
                progress = json.load(f)
            # Convert evaluated_questions back to a set
            progress['evaluated_questions'] = set(progress['evaluated_questions'])
            return progress
        
        return {
            'processed_files': {},
            'evaluated_questions': set()
        }
    
    def _save_progress(self):
        """
        Save the current progress to a JSON file.
        """
        # Convert set to list for JSON serialization
        progress_to_save = self.progress.copy()
        progress_to_save['evaluated_questions'] = list(self.progress['evaluated_questions'])
        
        with open(self.progress_file_path, 'w') as f:
            json.dump(progress_to_save, f, indent=4)
    
    def _generate_question_hash(self, question, filename): #_ is a convention to indicate that this method is private and should not be called directly by users of the class
        """
        Generate a unique hash for a question to track its evaluation status.
        
        Args:
            question (str): The question to hash
            filename (str): The filename containing the question
        
        Returns:
            str: Hash of the question and filename
        """
        # Combine question and filename to create a unique hash
        combined_string = f"{filename}_{question}"
        return hashlib.md5(combined_string.encode()).hexdigest()
    
    def is_file_processed(self, filename):
        """
        Check if a file has been fully processed.
        
        Args:
            filename (str): Name of the file to check
        
        Returns:
            bool: Whether the file has been fully processed
        """
        return self.progress['processed_files'].get(filename, False)
    
    def mark_file_processed(self, filename):
        """
        Mark a file as processed.
        
        Args:
            filename (str): Name of the file to mark
        """
        self.progress['processed_files'][filename] = True
        self._save_progress()
    
    def is_question_evaluated(self, question, filename):
        """
        Check if a question has already been manually evaluated.
        
        Args:
            question (str): The question to check
            filename (str): The filename containing the question
        
        Returns:
            bool: Whether the question has been evaluated
        """
        question_hash = self._generate_question_hash(question, filename)
        return question_hash in self.progress['evaluated_questions']

    def mark_question_evaluated(self, question, filename):
        """
        Mark a question as evaluated.
        
        Args:
            question (str): The question to mark
            filename (str): The filename containing the question
        """
        question_hash = self._generate_question_hash(question, filename)
        self.progress['evaluated_questions'].add(question_hash)
        self._save_progress()

    def _generate_question_hash(self, question, filename):
        """
        Generate a unique hash for a question to track its evaluation status.
        
        Args:
            question (str): The question to hash
            filename (str): The filename containing the question
        
        Returns:
            str: Hash of the question and filename
        """
        # Combine question and filename to create a unique hash
        combined_string = f"{filename}_{question}"
        return hashlib.md5(combined_string.encode()).hexdigest()
    
def manual_evaluate_entry(entry, tracker, filename):
    """
    Interactively evaluate a single entry with progress tracking.
    
    Args:
        entry (dict): A single JSON entry to be evaluated
        tracker (ManualEvaluationTracker): Progress tracking object
        filename (str): Name of the file containing the entry
    
    Returns:
        dict: Updated entry with manual evaluation results, or None if already evaluated
    """
    # Check if the question has already been evaluated
    if tracker.is_question_evaluated(entry['question'], filename):
        print(f"Question already evaluated: {entry['question']}")
        return None
    
    print("\n--- Manual Evaluation ---")
    print(f"Question: {entry['question']}")
    print(f"Model Response: {entry['response']}")
    print(f"Expected Answer: {entry['expected_answer'][0]}")
    
    while True:
        user_input = input("Is the response correct? (yes/no): ").lower().strip()
        if user_input in ['y', 'n', 'yes', 'no']:
            break
        print("Please enter 'yes' or 'no'.")
    
    # Create a new entry with manual evaluation
    manual_entry = entry.copy()
    manual_entry['manual_evaluation'] = user_input
    
    # If manually evaluated as incorrect, mark accordingly
    if user_input == 'no' or user_input == 'n':
        manual_entry['evaluation_status'] = 'manually_incorrect'
    
    # Mark the question as evaluated
    tracker.mark_question_evaluated(entry['question'], filename)
    
    return manual_entry

def process_json_results(input_directory, output_directory):
    """
    Process JSON result files with interactive manual evaluation and progress tracking.
    
    Args:
        input_directory (str): Directory containing input JSON files
        output_directory (str): Directory to save processed results
    """
    # Initialize progress tracker
    tracker = ManualEvaluationTracker(input_directory, output_directory)
    
    # Tracking variables for overall analysis
    overall_analysis = {
        'files_processed': 0,
        'total_entries': 0,
        'entries_needing_manual_evaluation': 0,
        'file_accuracies': {}
    }
    
    # Process each JSON file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('_results.json'):
            # Skip if file has already been fully processed
            if tracker.is_file_processed(filename):
                print(f"Skipping already processed file: {filename}")
                continue
            
            # Add this line to print the current file being processed
            print(30*"-")
            print(f"\nProcessing file: {filename}\n")
            print(30*"-")
            
            input_filepath = os.path.join(input_directory, filename)
            
            # Read the input JSON file
            with open(input_filepath, 'r') as f:
                results = json.load(f)
            
            # Filter out entries that need manual evaluation
            unevaluated_results = [
                entry for entry in results 
                if entry.get('self_evaluation', '').lower() == 'no'
            ]
            
            # If there are unevaluated results, process them
            if unevaluated_results:
                overall_analysis['files_processed'] += 1
                overall_analysis['total_entries'] += len(results)
                overall_analysis['entries_needing_manual_evaluation'] += len(unevaluated_results)
                
                # Manually evaluate each unevaluated entry
                manually_evaluated_results = []
                for entry in unevaluated_results:
                    # Skip if already evaluated
                    manually_evaluated_entry = manual_evaluate_entry(entry, tracker, filename)
                    if manually_evaluated_entry:
                        manually_evaluated_results.append(manually_evaluated_entry)                
                # Calculate accuracy for this file
                file_analysis = {
                    'total_entries': len(results),
                    'manually_evaluated_entries': len(manually_evaluated_results),
                    'correct_entries': 0,
                    'accuracy_percentage': 0
                }
                
                if manually_evaluated_results:
                    # Count correct entries
                    file_analysis['correct_entries'] = sum(
                        1 for entry in manually_evaluated_results 
                        if entry.get('manual_evaluation', '') in ['yes', 'y']
                    )
                    # Calculate accuracy
                    file_analysis['accuracy_percentage'] = (
                        file_analysis['correct_entries'] / len(manually_evaluated_results) * 100
                    )
                
                # Store file accuracy
                overall_analysis['file_accuracies'][filename] = file_analysis
                
                # Create output filename
                output_filename = f'manual_evaluation_{filename}'
                output_filepath = os.path.join(output_directory, output_filename)
                
                # Save manually evaluated results to a new file
                if manually_evaluated_results:
                    with open(output_filepath, 'w') as f:
                        json.dump(manually_evaluated_results, f, indent=4)
                
                # Mark file as processed
                tracker.mark_file_processed(filename)
                
                print(f"\nProcessed {filename}:")
                print(f"Total entries: {len(results)}")
                print(f"Manually evaluated entries: {len(manually_evaluated_results)}")
                print(f"Accuracy: {file_analysis['accuracy_percentage']:.2f}%")
    
    # Calculate overall accuracy with error handling
    accurate_files = [
        file_data['accuracy_percentage'] 
        for file_data in overall_analysis['file_accuracies'].values() 
        if file_data['manually_evaluated_entries'] > 0
    ]
    
    if accurate_files:
        overall_analysis['overall_accuracy_percentage'] = sum(accurate_files) / len(accurate_files)
    else:
        overall_analysis['overall_accuracy_percentage'] = 0
    
    # Save overall analysis report
    analysis_filepath = os.path.join(output_directory, 'analysis_report.json')
    with open(analysis_filepath, 'w') as f:
        json.dump(overall_analysis, f, indent=4)
    
    # Print final summary
    print("\n--- Final Analysis ---")
    print(f"Total files processed: {overall_analysis['files_processed']}")
    print(f"Total entries: {overall_analysis['total_entries']}")
    print(f"Entries needing manual evaluation: {overall_analysis['entries_needing_manual_evaluation']}")
    print(f"Overall Accuracy: {overall_analysis['overall_accuracy_percentage']:.2f}%")

def main():
    # Example usage
    input_dir = './JSON_output'  # Directory containing your JSON files
    output_dir = './JSON_output/JSON_analysis'  # Directory to save processed files
    
    process_json_results(input_dir, output_dir)

if __name__ == '__main__':
    main()