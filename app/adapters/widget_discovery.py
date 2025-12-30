"""
Widget Discovery Module
Automatically discovers Orange3 widgets by scanning widget directories
and parsing Python files using AST (no import required).
"""

import os
import ast
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path


# Category color definitions from Orange3
CATEGORY_COLORS = {
    "Data": "#FFD39F",
    "Transform": "#FF9D5E",
    "Visualize": "#FFB7B1",
    "Model": "#FAC1D9",
    "Evaluate": "#C3F3F3",
    "Unsupervised": "#CAE1EF",
    "Text Mining": "#B8E0D2",  # Light teal for text mining
}

# Category priorities
CATEGORY_PRIORITIES = {
    "Data": 1,
    "Transform": 2,
    "Visualize": 3,
    "Model": 4,
    "Evaluate": 5,
    "Unsupervised": 6,
    "Text Mining": 7,
}


class WidgetDiscovery:
    """Discovers Orange3 widgets from the filesystem using AST parsing."""
    
    def __init__(self, orange3_path: Optional[str] = None, orange3_text_path: Optional[str] = None):
        """
        Initialize the widget discovery.
        
        Args:
            orange3_path: Path to Orange3 installation. If None, tries to find it.
            orange3_text_path: Path to Orange3-Text installation. If None, tries to find it.
        """
        self.orange3_path = orange3_path or self._find_orange3_path()
        self.orange3_text_path = orange3_text_path or self._find_orange3_text_path()
        self.categories: Dict[str, Dict] = {}
        self.widgets: List[Dict] = []
        
    def _find_orange3_path(self) -> Optional[str]:
        """Try to find Orange3 installation path."""
        # Common locations to check
        possible_paths = [
            # Local development
            os.path.expanduser("~/works/test/orange3/orange3/Orange/widgets"),
            # pip installed
            os.path.join(os.path.dirname(__file__), "..", "..", "..", ".venv", "lib", "python3.11", "site-packages", "Orange", "widgets"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_orange3_text_path(self) -> Optional[str]:
        """Try to find Orange3-Text installation path."""
        # Common locations to check
        possible_paths = [
            # Local development
            os.path.expanduser("~/works/test/orange3/orange3-text/orangecontrib/text/widgets"),
            # pip installed
            os.path.join(os.path.dirname(__file__), "..", "..", "..", ".venv", "lib", "python3.11", "site-packages", "orangecontrib", "text", "widgets"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def discover(self) -> Dict[str, Any]:
        """
        Discover all widgets and categories.
        
        Returns:
            Dictionary with 'categories' and 'widgets' keys.
        """
        self.categories = {}
        self.widgets = []
        
        # Discover Orange3 core widgets
        if self.orange3_path and os.path.exists(self.orange3_path):
            self._discover_orange3_widgets()
        else:
            print(f"Warning: Orange3 widgets path not found: {self.orange3_path}")
        
        # Discover Orange3-Text widgets
        if self.orange3_text_path and os.path.exists(self.orange3_text_path):
            self._discover_orange3_text_widgets()
        else:
            print(f"Warning: Orange3-Text widgets path not found: {self.orange3_text_path}")
        
        if not self.widgets:
            return {"categories": [], "widgets": []}
        
        return self._format_result()
    
    def _discover_orange3_widgets(self):
        """Discover Orange3 core widgets."""
        # Scan widget directories
        widget_dirs = ['data', 'visualize', 'model', 'evaluate', 'unsupervised']
        
        for subdir in widget_dirs:
            dir_path = os.path.join(self.orange3_path, subdir)
            if not os.path.exists(dir_path):
                continue
            
            # Read category info from __init__.py
            cat_info = self._read_category_info(dir_path, subdir)
            
            # Scan widget files
            self._scan_widget_directory(dir_path, subdir, cat_info)
    
    def _discover_orange3_text_widgets(self):
        """Discover Orange3-Text widgets."""
        cat_info = {
            'name': 'Text Mining',
            'background': CATEGORY_COLORS.get('Text Mining', '#B8E0D2'),
            'priority': CATEGORY_PRIORITIES.get('Text Mining', 7)
        }
        
        # Scan widget files in the text widgets directory
        self._scan_widget_directory(self.orange3_text_path, 'text', cat_info, icon_prefix='text/')
    
    def _scan_widget_directory(self, dir_path: str, subdir: str, cat_info: Dict, icon_prefix: str = ''):
        """Scan a widget directory and add widgets to the discovery result."""
        for filename in sorted(os.listdir(dir_path)):
            if filename.startswith('ow') and filename.endswith('.py'):
                filepath = os.path.join(dir_path, filename)
                widget_info = self._extract_widget_info(filepath, subdir)
                
                if widget_info and widget_info.get('name'):
                    # Add special inputs for Evaluate widgets (Learner, Model)
                    self._add_evaluate_widget_inputs(widget_info)
                    # Add inputs/outputs for Text Mining widgets
                    self._add_text_mining_widget_io(widget_info)
                    # Determine category (widget may override)
                    widget_category = widget_info.get('category') or cat_info['name']
                    
                    # Ensure category exists
                    if widget_category not in self.categories:
                        self.categories[widget_category] = {
                            'name': widget_category,
                            'background': CATEGORY_COLORS.get(widget_category, cat_info['background']),
                            'priority': CATEGORY_PRIORITIES.get(widget_category, 10),
                            'widgets': []
                        }
                    
                    # Handle icon path for text mining widgets
                    icon = widget_info.get('icon', 'icons/Unknown.svg')
                    if icon_prefix and not icon.startswith('http'):
                        # Add prefix for text mining icons
                        icon = icon_prefix + icon.replace('icons/', '')
                    
                    # Add widget to category
                    # Sort ports to prioritize Data type (most commonly used)
                    inputs = self._sort_ports_by_priority(widget_info.get('inputs', []))
                    outputs = self._sort_ports_by_priority(widget_info.get('outputs', []))
                    
                    widget_data = {
                        'id': self._generate_widget_id(widget_info['name']),
                        'name': widget_info['name'],
                        'description': widget_info.get('description', ''),
                        'icon': icon,
                        'category': widget_category,
                        'priority': widget_info.get('priority', 9999),
                        'inputs': inputs,
                        'outputs': outputs,
                        'keywords': widget_info.get('keywords', []),
                        'source': 'orange3-text' if icon_prefix else 'orange3',
                    }
                    
                    self.categories[widget_category]['widgets'].append(widget_data)
                    self.widgets.append(widget_data)
    
    def _read_category_info(self, dir_path: str, default_name: str) -> Dict:
        """Read category info from __init__.py."""
        init_file = os.path.join(dir_path, '__init__.py')
        cat_name = default_name.capitalize()
        cat_bg = CATEGORY_COLORS.get(cat_name, '#999999')
        cat_priority = CATEGORY_PRIORITIES.get(cat_name, 10)
        
        if os.path.exists(init_file):
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                
                # Parse using regex (simple key-value pairs)
                name_match = re.search(r'NAME\s*=\s*["\']([^"\']+)["\']', content)
                bg_match = re.search(r'BACKGROUND\s*=\s*["\']([^"\']+)["\']', content)
                priority_match = re.search(r'PRIORITY\s*=\s*(\d+)', content)
                
                if name_match:
                    cat_name = name_match.group(1)
                if bg_match:
                    cat_bg = bg_match.group(1)
                if priority_match:
                    cat_priority = int(priority_match.group(1))
                    
            except Exception as e:
                print(f"Warning: Error reading {init_file}: {e}")
        
        return {
            'name': cat_name,
            'background': cat_bg,
            'priority': cat_priority
        }
    
    def _extract_widget_info(self, filepath: str, category: str) -> Optional[Dict]:
        """Extract widget info from Python file using AST parsing."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            info = {
                'name': None,
                'description': None,
                'icon': None,
                'category': None,
                'priority': 9999,  # Default high priority (shown last)
                'inputs': [],
                'outputs': [],
                'keywords': [],
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a widget class
                    if not self._is_widget_class(node):
                        continue
                    
                    # Get inherited inputs/outputs from base classes
                    inherited_io = self._get_inherited_io(node)
                    
                    # Extract class-level attributes and nested classes
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            self._extract_assign(item, info)
                        elif isinstance(item, ast.AnnAssign):
                            self._extract_ann_assign(item, info)
                        elif isinstance(item, ast.ClassDef):
                            # Handle nested Inputs/Outputs classes
                            if item.name == 'Inputs':
                                info['inputs'] = self._extract_io_class(item, 'Input')
                            elif item.name == 'Outputs':
                                info['outputs'] = self._extract_io_class(item, 'Output')
                    
                    # Merge inherited IO with extracted IO (inherited first, then widget-specific)
                    # Use reversed() to maintain original order when inserting at position 0
                    if inherited_io['inputs']:
                        existing_ids = {p['id'] for p in info['inputs']}
                        for inp in reversed(inherited_io['inputs']):
                            if inp['id'] not in existing_ids:
                                info['inputs'].insert(0, inp)
                    
                    if inherited_io['outputs']:
                        existing_ids = {p['id'] for p in info['outputs']}
                        for outp in reversed(inherited_io['outputs']):
                            if outp['id'] not in existing_ids:
                                info['outputs'].insert(0, outp)
                    
                    if info['name']:
                        return info
            
            return None
            
        except Exception as e:
            print(f"Warning: Error parsing {filepath}: {e}")
            return None
    
    def _is_widget_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is a widget class."""
        for base in node.bases:
            base_name = ''
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            
            if 'Widget' in base_name or base_name.startswith('OW'):
                return True
        return False
    
    def _get_inherited_io(self, node: ast.ClassDef) -> Dict[str, List[Dict]]:
        """Get inherited inputs/outputs from known base classes."""
        inherited = {'inputs': [], 'outputs': []}
        
        for base in node.bases:
            base_name = ''
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            
            # Data projection widgets (scatter plot, MDS, t-SNE, etc.)
            if 'DataProjectionWidget' in base_name:
                inherited['inputs'].append({'id': 'data', 'name': 'Data', 'type': 'Data'})
                inherited['inputs'].append({'id': 'data_subset', 'name': 'Data Subset', 'type': 'Data'})
                inherited['outputs'].append({'id': 'selected_data', 'name': 'Selected Data', 'type': 'Data'})
                inherited['outputs'].append({'id': 'annotated_data', 'name': 'Data', 'type': 'Data'})
            
            # Report widgets
            elif 'Report' in base_name and 'Widget' in base_name:
                pass  # No default IO
            
            # Learner widgets (OWBaseLearner, OWLearnerWidget, etc.)
            elif 'Learner' in base_name or 'BaseLearner' in base_name:
                inherited['inputs'].append({'id': 'data', 'name': 'Data', 'type': 'Data'})
                inherited['inputs'].append({'id': 'preprocessor', 'name': 'Preprocessor', 'type': 'Preprocessor'})
                inherited['outputs'].append({'id': 'learner', 'name': 'Learner', 'type': 'Learner'})
                inherited['outputs'].append({'id': 'model', 'name': 'Model', 'type': 'Model'})
        
        return inherited
    
    def _add_evaluate_widget_inputs(self, widget_info: Dict) -> None:
        """Add Learner/Model inputs for Evaluate category widgets that need them."""
        widget_name = widget_info.get('name', '').lower()
        inputs = widget_info.get('inputs', [])
        input_ids = {inp.get('id') for inp in inputs}
        
        # Test and Score widget needs Learner input
        if 'test' in widget_name and 'score' in widget_name:
            if 'learner' not in input_ids:
                # Insert Learner input after Data, before Preprocessor
                learner_input = {'id': 'learner', 'name': 'Learner', 'type': 'Learner', 'multiple': True}
                # Find position after Data
                insert_pos = 0
                for i, inp in enumerate(inputs):
                    if inp.get('type') == 'Data':
                        insert_pos = i + 1
                inputs.insert(insert_pos, learner_input)
        
        # Predictions widget needs Predictors (Model) input
        elif 'prediction' in widget_name:
            if 'predictors' not in input_ids and 'model' not in input_ids:
                # Insert Predictors (Model) input after Data
                predictors_input = {'id': 'predictors', 'name': 'Predictors', 'type': 'Model', 'multiple': True}
                insert_pos = 0
                for i, inp in enumerate(inputs):
                    if inp.get('type') == 'Data':
                        insert_pos = i + 1
                inputs.insert(insert_pos, predictors_input)
    
    def _add_text_mining_widget_io(self, widget_info: Dict) -> None:
        """Add inputs/outputs for Text Mining widgets that need manual configuration."""
        widget_name = widget_info.get('name', '').lower()
        inputs = widget_info.get('inputs', [])
        outputs = widget_info.get('outputs', [])
        input_ids = {inp.get('id') for inp in inputs}
        output_ids = {outp.get('id') for outp in outputs}
        
        # Corpus widget
        if widget_name == 'corpus':
            if 'data' not in input_ids:
                inputs.insert(0, {'id': 'data', 'name': 'Data', 'type': 'Data'})
            if 'corpus' not in output_ids:
                outputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
        
        # Preprocess Text widget
        elif 'preprocess' in widget_name and 'text' in widget_name:
            if 'corpus' not in input_ids:
                inputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'corpus' not in output_ids:
                outputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
        
        # Bag of Words widget
        elif 'bag' in widget_name and 'words' in widget_name:
            if 'corpus' not in input_ids:
                inputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'corpus' not in output_ids:
                outputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'bow' not in output_ids and 'data' not in output_ids:
                outputs.append({'id': 'bow', 'name': 'Bag of Words', 'type': 'Data'})
        
        # Word Cloud widget
        elif 'word' in widget_name and 'cloud' in widget_name:
            # Inputs: Corpus (default), Topic (optional)
            if 'corpus' not in input_ids:
                inputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'topic' not in input_ids:
                inputs.append({'id': 'topic', 'name': 'Topic', 'type': 'Topic'})
            # Outputs: Corpus (default), Selected Words, Word Counts
            if 'corpus' not in output_ids:
                outputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'selected_words' not in output_ids:
                outputs.append({'id': 'selected_words', 'name': 'Selected Words', 'type': 'Data'})
            if 'word_counts' not in output_ids:
                outputs.append({'id': 'word_counts', 'name': 'Word Counts', 'type': 'Data'})
        
        # Corpus Viewer widget
        elif 'corpus' in widget_name and 'viewer' in widget_name:
            if 'corpus' not in input_ids:
                inputs.insert(0, {'id': 'corpus', 'name': 'Corpus', 'type': 'Corpus'})
            if 'selected_documents' not in output_ids:
                outputs.insert(0, {'id': 'selected_documents', 'name': 'Selected Documents', 'type': 'Corpus'})
        
        # Word List widget
        elif 'word' in widget_name and 'list' in widget_name:
            if 'words' not in output_ids:
                outputs.insert(0, {'id': 'words', 'name': 'Words', 'type': 'Word List'})
        
        # Word Enrichment widget
        elif 'word' in widget_name and 'enrichment' in widget_name:
            if 'data' not in input_ids:
                inputs.insert(0, {'id': 'data', 'name': 'Data', 'type': 'Data'})
            if 'words' not in input_ids:
                inputs.append({'id': 'words', 'name': 'Words', 'type': 'Word List'})
            if 'selected_words' not in output_ids:
                outputs.insert(0, {'id': 'selected_words', 'name': 'Selected Words', 'type': 'Data'})
        
        widget_info['inputs'] = inputs
        widget_info['outputs'] = outputs
    
    def _extract_assign(self, item: ast.Assign, info: Dict):
        """Extract info from simple assignment."""
        for target in item.targets:
            if isinstance(target, ast.Name):
                name = target.id
                value = self._get_constant_value(item.value)
                
                if name == 'name' and value:
                    info['name'] = value
                elif name == 'description' and value:
                    info['description'] = value
                elif name == 'icon' and value:
                    info['icon'] = value
                elif name == 'category' and value:
                    info['category'] = value
                elif name == 'priority':
                    # Priority is an integer
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, (int, float)):
                        info['priority'] = int(item.value.value)
                    # Python 3.7 compatibility (ast.Num was deprecated in 3.8, removed in 3.12)
                    elif hasattr(ast, 'Num') and isinstance(item.value, ast.Num):
                        info['priority'] = item.value.n
                elif name == 'keywords' and isinstance(item.value, ast.List):
                    info['keywords'] = [self._get_constant_value(e) for e in item.value.elts if self._get_constant_value(e)]
    
    def _extract_ann_assign(self, item: ast.AnnAssign, info: Dict):
        """Extract info from annotated assignment (Inputs/Outputs)."""
        if not isinstance(item.target, ast.Name):
            return
        
        name = item.target.id
        
        # Check for Inputs/Outputs class definitions
        if name == 'Inputs' and isinstance(item.value, ast.Call):
            # This is for newer Orange3 style: Inputs = Inputs(...)
            pass
        elif name == 'Outputs' and isinstance(item.value, ast.Call):
            pass
    
    def _extract_io_class(self, class_node: ast.ClassDef, io_type: str) -> List[Dict]:
        """
        Extract inputs or outputs from nested Inputs/Outputs class.
        
        Orange3 widgets define inputs/outputs like:
            class Inputs:
                data = Input("Data", Table)
            
            class Outputs:
                data = Output("Data", Table)
        """
        ports = []
        
        for item in class_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        port_id = target.id
                        port_info = self._parse_io_call(item.value, port_id, io_type)
                        if port_info:
                            ports.append(port_info)
        
        return ports
    
    # Known Orange3 constants for signal names
    KNOWN_SIGNAL_CONSTANTS = {
        'ANNOTATED_DATA_SIGNAL_NAME': 'Data',
        'DOMAIN_ROLE_HINTS': 'Domain Role Hints',
    }
    
    def _parse_io_call(self, node, port_id: str, io_type: str) -> Optional[Dict]:
        """
        Parse Input(...) or Output(...) call.
        
        Format: Input("Name", Type, ...) or Output("Name", Type, ...)
        """
        if not isinstance(node, ast.Call):
            return None
        
        # Get the function name
        func_name = ''
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Only process Input or Output calls
        if func_name not in ('Input', 'Output'):
            return None
        
        # Get the port name (first argument)
        port_name = port_id  # default to variable name
        if node.args and len(node.args) >= 1:
            first_arg = node.args[0]
            # Check if it's a string constant
            name_val = self._get_constant_value(first_arg)
            if name_val:
                port_name = name_val
            # Check if it's a known constant variable
            elif isinstance(first_arg, ast.Name) and first_arg.id in self.KNOWN_SIGNAL_CONSTANTS:
                port_name = self.KNOWN_SIGNAL_CONSTANTS[first_arg.id]
            # Fallback: convert variable name to readable format
            elif port_name == port_id:
                port_name = self._format_port_name(port_id)
        
        # Get the type (second argument)
        port_type = 'Data'  # default
        if node.args and len(node.args) >= 2:
            type_node = node.args[1]
            if isinstance(type_node, ast.Name):
                port_type = self._simplify_type_name(type_node.id)
            elif isinstance(type_node, ast.Attribute):
                port_type = self._simplify_type_name(type_node.attr)
        
        return {
            'id': port_id,
            'name': port_name,
            'type': port_type
        }
    
    def _format_port_name(self, port_id: str) -> str:
        """Convert port_id to readable name (e.g., 'annotated_data' -> 'Annotated Data')."""
        # Replace underscores with spaces and capitalize each word
        return ' '.join(word.capitalize() for word in port_id.split('_'))
    
    def _simplify_type_name(self, type_name: str) -> str:
        """Simplify Orange3 type names for display."""
        type_map = {
            'Table': 'Data',
            'Domain': 'Data',
            'Learner': 'Learner',
            'Model': 'Model',
            'DistMatrix': 'Distance',
            'Preprocess': 'Preprocessor',
            'Results': 'Evaluation Results',
            'Network': 'Network',
            # Text Mining types
            'Corpus': 'Corpus',
            'Topic': 'Topic',
            'WordList': 'Word List',
            'Document': 'Document',
        }
        return type_map.get(type_name, type_name)
    
    def _sort_ports_by_priority(self, ports: List[Dict]) -> List[Dict]:
        """
        Sort ports to prioritize Data type (most commonly connected).
        Priority order: Data > Corpus > other types
        """
        if not ports:
            return ports
        
        # Define type priority (lower = higher priority)
        type_priority = {
            'Data': 0,
            'Corpus': 1,
            'Learner': 2,
            'Model': 3,
        }
        
        def get_priority(port):
            port_type = port.get('type', '')
            return type_priority.get(port_type, 99)
        
        return sorted(ports, key=get_priority)
    
    def _get_constant_value(self, node) -> Optional[str]:
        """Get constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        # Python 3.7 compatibility (ast.Str was deprecated in 3.8, removed in 3.12)
        if hasattr(ast, 'Str') and isinstance(node, ast.Str):
            return node.s
        return None
    
    def _generate_widget_id(self, name: str) -> str:
        """Generate a URL-friendly widget ID from name."""
        # Convert to lowercase, replace spaces with hyphens
        widget_id = name.lower()
        widget_id = re.sub(r'[^a-z0-9]+', '-', widget_id)
        widget_id = widget_id.strip('-')
        return widget_id
    
    def _format_result(self) -> Dict[str, Any]:
        """Format the discovery result."""
        # Sort categories by priority
        sorted_categories = sorted(
            self.categories.values(),
            key=lambda c: c.get('priority', 10)
        )
        
        # Format for frontend
        formatted_categories = []
        for cat in sorted_categories:
            if not cat['widgets']:
                continue
            
            # Sort widgets by priority (same as Orange3)
            sorted_widgets = sorted(cat['widgets'], key=lambda w: (w.get('priority', 9999), w['name']))
            
            formatted_categories.append({
                'name': cat['name'],
                'color': cat['background'],
                'priority': cat['priority'],
                'widgets': sorted_widgets
            })
        
        return {
            'categories': formatted_categories,
            'widgets': self.widgets,
            'total': len(self.widgets)
        }


# Singleton instance
_discovery_instance: Optional[WidgetDiscovery] = None


def get_widget_discovery(orange3_path: Optional[str] = None) -> WidgetDiscovery:
    """Get or create the widget discovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = WidgetDiscovery(orange3_path)
    return _discovery_instance


def discover_widgets(orange3_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Discover all Orange3 widgets.
    
    Args:
        orange3_path: Optional path to Orange3 widgets directory.
        
    Returns:
        Dictionary with categories and widgets.
    """
    discovery = get_widget_discovery(orange3_path)
    return discovery.discover()


if __name__ == '__main__':
    # Test discovery
    result = discover_widgets()
    print(json.dumps(result, indent=2, default=str))

