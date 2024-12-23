import json
import os
from typing import Set

class EvaluationTracker:
    def __init__(self, tracking_file: str = "evaluation_progress.json"):
        self.tracking_file = tracking_file
        self.completed_items: Set[str] = set()
        self._load_progress()
    
    def _load_progress(self):
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                data = json.load(f)
                self.completed_items = set(data.get('completed_items', []))
    
    def save_progress(self):
        with open(self.tracking_file, 'w') as f:
            json.dump({
                'completed_items': list(self.completed_items)
            }, f)
    
    def mark_completed(self, file_path: str, question: str):
        item_id = f"{file_path}:{question}"
        self.completed_items.add(item_id)
        self.save_progress()
    
    def is_completed(self, file_path: str, question: str) -> bool:
        item_id = f"{file_path}:{question}"
        return item_id in self.completed_items