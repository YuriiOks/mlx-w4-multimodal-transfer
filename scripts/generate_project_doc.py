"""
✨ Enhanced Project Documentation Generator ✨

This script creates comprehensive documentation with consistent content
between Markdown and HTML outputs.

Features:
- Directory tree with customizable exclusions
- File content with syntax highlighting
- Module documentation with docstrings
- Project statistics
- Consistent content between Markdown and HTML outputs
"""

import os
import re
import datetime
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import markdown
from collections import Counter
import shutil

# ----- Constants and Configuration -----

# Common directories to exclude by default
DEFAULT_EXCLUDE_DIRS = [
    # Virtual environments
    '.venv', 'venv', 'env', 'virtualenv', '.virtualenv',
    # Build and distribution
    'build', 'dist', 'site-packages', 'node_modules',
    # Cache directories
    '__pycache__', '.pytest_cache', '.mypy_cache', '.coverage', 
    '.tox', '.nox', 'htmlcov', '__pypackages__', '.eggs', '.egg-info',
    # Version control
    '.git', '.hg', '.svn', 
    # IDE directories
    '.idea', '.vs', '.vscode',
    # Misc
    '.ipynb_checkpoints', 'wandb', '.wandb'
]

# File patterns to exclude by default
DEFAULT_EXCLUDE_PATTERNS = [
    # Compiled files
    r'.*\.pyc$', r'.*\.pyo$', r'.*\.pyd$', r'.*\.so$', 
    # Binary and data files
    r'.*\.bin$', r'.*\.dat$', r'.*\.pkl$', r'.*\.db$',
    # Temporary and system files
    r'.*\.DS_Store$', r'.*\.tmp$', r'.*\.bak$', r'.*\.swp$',
    # Log files
    r'.*\.log$', 
    # Large generated files
    r'.*\.egg-info$', r'.*\.coverage$', r'.*\.cache$'
]

# File extensions to programming language mapping
LANGUAGE_MAP = {
    # Python
    '.py': 'python',
    '.pyx': 'python',
    '.pyi': 'python',
    # Web
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.js': 'javascript',
    '.jsx': 'jsx',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    # Data
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.xml': 'xml',
    '.csv': 'csv',
    # Shell
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'bash',
    '.fish': 'fish',
    '.bat': 'batch',
    '.cmd': 'batch',
    '.ps1': 'powershell',
    # Config
    '.ini': 'ini',
    '.cfg': 'ini',
    '.conf': 'ini',
    '.env': 'ini',
    # Markdown and documentation
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.rst': 'rst',
    '.txt': 'text',
    # Docker, etc.
    '.dockerfile': 'dockerfile',
    '.Dockerfile': 'dockerfile'
}

# Special filenames to language mapping
FILENAME_MAP = {
    'Dockerfile': 'dockerfile',
    'docker-compose.yml': 'yaml',
    'docker-compose.yaml': 'yaml',
    'Makefile': 'makefile',
    'CMakeLists.txt': 'cmake',
    'requirements.txt': 'text',
    'setup.py': 'python',
    'setup.cfg': 'ini',
    'pyproject.toml': 'toml',
    'package.json': 'json',
    'tsconfig.json': 'json',
    '.gitignore': 'text',
    '.gitattributes': 'text',
    '.gitconfig': 'text'
}

# ----- Helper Functions -----

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    current_file = Path(__file__).resolve()
    # Assuming this script is in a scripts/ or tools/ subdirectory
    # Try to find the project root (with .git, pyproject.toml, etc.)
    current_dir = current_file.parent
    
    # Check if we're already at the project root
    if is_project_root(current_dir):
        return current_dir
    
    # Move up until we find a project root indicator
    parent_dir = current_dir.parent
    while parent_dir != parent_dir.parent:  # Stop at filesystem root
        if is_project_root(parent_dir):
            return parent_dir
        parent_dir = parent_dir.parent
    
    # If no clear project root found, default to parent of script
    return current_file.parent.parent

def is_project_root(path: Path) -> bool:
    """Check if the given path is likely to be a project root."""
    root_indicators = [
        '.git',
        'pyproject.toml',
        'setup.py',
        'package.json',
        'Cargo.toml',  # Rust
        'go.mod',      # Go
        'pom.xml',     # Maven/Java
        'build.gradle' # Gradle/Java
    ]
    return any((path / indicator).exists() for indicator in root_indicators)

def get_file_language(file_path: Path) -> str:
    """Determine the language of a file based on its extension or filename."""
    # First check if the filename itself has a special mapping
    if file_path.name in FILENAME_MAP:
        return FILENAME_MAP[file_path.name]
    
    # Then check the extension
    extension = file_path.suffix.lower()
    return LANGUAGE_MAP.get(extension, 'text')

def get_file_content(file_path: Path, max_size: int = 500000) -> str:
    """
    Extract the content of a file, with size limits for safety.
    
    Args:
        file_path: Path to the file
        max_size: Maximum file size in bytes (default: 500KB)
        
    Returns:
        File content as string or error message
    """
    try:
        # Check file size first
        file_size = file_path.stat().st_size
        if file_size > max_size:
            return f"File too large to display: {file_size // 1024} KB (limit: {max_size // 1024} KB)"
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        return "File contains binary data or encoding issues"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def extract_docstring(file_path: Path) -> str:
    """
    Extract the module-level docstring from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Extracted docstring or a message if none found
    """
    try:
        content = get_file_content(file_path)
        # Use regex to find the module-level docstring
        # Look for triple quotes after possible comments and whitespace
        pattern = r'(?:^|\s)(?:(?:\'\'\'|""")(.+?)(?:\'\'\'|"""))(?:\s|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            docstring = match.group(1).strip()
            return docstring
        else:
            # Simpler pattern for single-line docstrings
            pattern = r'(?:^|\s)(?:(?:\'\'\'|""")(.+?)(?:\'\'\'|"""))'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
            
        return "No docstring found."
    except Exception as e:
        return f"Error extracting docstring: {str(e)}"

# ----- Core Functionality -----

def generate_directory_tree(
    root_dir: Path,
    exclude_dirs: List[str] = None,
    exclude_patterns: List[str] = None,
    max_depth: int = None
) -> str:
    """
    Generate a directory tree structure as a string.
    
    Args:
        root_dir: Path to the root directory
        exclude_dirs: List of directory names to exclude
        exclude_patterns: List of regex patterns to exclude
        max_depth: Maximum depth to traverse (None for unlimited)
        
    Returns:
        Formatted directory tree as a string
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
    
    # Convert patterns to compiled regex objects
    exclude_patterns = [re.compile(pattern) for pattern in exclude_patterns]
    
    tree_str = []
    root_dir = Path(root_dir)
    
    def should_exclude(path: Path) -> bool:
        """Determine if a path should be excluded from the tree."""
        # Check if directory name is in exclude list
        if path.name in exclude_dirs:
            return True
        
        # Check if any parent directory is in exclude_dirs
        for parent in path.parents:
            if parent.name in exclude_dirs:
                return True
        
        # Check exclude patterns against the full path
        for pattern in exclude_patterns:
            if pattern.match(str(path)):
                return True
        
        return False
    
    def add_directory(directory: Path, prefix: str = '', depth: int = 0) -> None:
        """Recursively add directory and its contents to the tree."""
        # Check max depth
        if max_depth is not None and depth > max_depth:
            tree_str.append(f"{prefix}└── ... (max depth reached)")
            return
        
        # Add directory name
        tree_str.append(f"{prefix}📁 {directory.name}/")
        
        # Get all items in directory
        try:
            # First get all subdirectories, then files, each sorted alphabetically
            dirs = sorted([item for item in directory.iterdir() if item.is_dir() and not should_exclude(item)])
            files = sorted([item for item in directory.iterdir() if item.is_file() and not should_exclude(item)])
            
            # Process all items (directories first, then files)
            items = dirs + files
            
            # Process each item
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                
                if item.is_dir():
                    # Add directory and its contents
                    next_prefix = prefix + ('└── ' if is_last else '├── ')
                    continuation_prefix = prefix + ('    ' if is_last else '│   ')
                    add_directory(item, next_prefix, depth + 1)
                else:
                    # Add file
                    tree_str.append(f"{prefix}{'└── ' if is_last else '├── '}📄 {item.name}")
        
        except PermissionError:
            tree_str.append(f"{prefix}  ⚠️ Permission denied")
            return
    
    # Start with the root directory
    add_directory(root_dir)
    return '\n'.join(tree_str)

def generate_module_documentation(root_dir: Path, exclude_dirs: List[str] = None) -> str:
    """
    Generate documentation for Python modules in the project.
    
    Args:
        root_dir: Path to the project root
        exclude_dirs: List of directory names to exclude
        
    Returns:
        Module documentation as a string
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    root_dir = Path(root_dir)
    module_docs = []
    
    # Find all Python files
    python_files = []
    for path in root_dir.glob('**/*.py'):
        # Skip files in excluded directories
        if any(excluded in str(path) for excluded in exclude_dirs):
            continue
        python_files.append(path)
    
    # Sort files by path for consistent output
    python_files.sort()
    
    for py_file in python_files:
        relative_path = py_file.relative_to(root_dir)
        docstring = extract_docstring(py_file)
        
        # Create a header with the file path
        module_docs.append(f"### {relative_path}\n\n{docstring}\n")
    
    if not module_docs:
        return "No Python modules found in the project."
    
    return "\n".join(module_docs)

def generate_file_content_documentation(
    root_dir: Path,
    include_patterns: List[str] = None,
    exclude_dirs: List[str] = None,
    max_file_count: int = 20
) -> str:
    """
    Generate documentation including full content of important files.
    
    Args:
        root_dir: Path to the project root
        include_patterns: List of patterns for files to include
        exclude_dirs: List of directories to exclude
        max_file_count: Maximum number of files to include
        
    Returns:
        File content documentation as a string
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    if include_patterns is None:
        # Default to README files and a few important configuration files
        include_patterns = [
            r'README\.md$',
            r'requirements\.txt$',
            r'setup\.py$',
            r'pyproject\.toml$',
            r'.*\.py$'  # Include Python files
        ]
    
    # Compile the patterns
    include_patterns = [re.compile(pattern) for pattern in include_patterns]
    
    root_dir = Path(root_dir)
    content_docs = []
    file_count = 0
    
    # First, handle README.md specially
    readme_path = root_dir / "README.md"
    if readme_path.exists():
        content_docs.append("## README\n")
        content_docs.append("```markdown")
        content_docs.append(get_file_content(readme_path))
        content_docs.append("```\n")
        file_count += 1
    
    # Next, find core Python files
    py_files = []
    
    # Priority files (in root or src directory)
    for py_file in list(root_dir.glob("*.py")) + list((root_dir / "src").glob("*.py") if (root_dir / "src").exists() else []):
        if py_file.name not in ["__init__.py"] and file_count < max_file_count:
            py_files.append(py_file)
            file_count += 1
    
    # Additional important files
    for pattern in include_patterns:
        if file_count >= max_file_count:
            break
        
        for file_path in root_dir.glob("**/*"):
            if file_count >= max_file_count:
                break
                
            # Skip directories and excluded directories
            if file_path.is_dir() or any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
                
            # Skip files already in py_files
            if file_path in py_files:
                continue
                
            # Check if file matches one of our include patterns
            if any(pattern.search(file_path.name) for pattern in include_patterns):
                py_files.append(file_path)
                file_count += 1
    
    # Sort files by name for consistent documentation
    py_files.sort()
    
    # Add file content to documentation
    if py_files:
        content_docs.append("## Core Files\n")
        
        for file_path in py_files:
            # Skip files in or under excluded directories
            if any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
                
            # Get file's relative path and language for syntax highlighting
            relative_path = file_path.relative_to(root_dir)
            language = get_file_language(file_path)
            
            content_docs.append(f"### {relative_path}\n")
            content_docs.append(f"```{language}")
            content_docs.append(get_file_content(file_path))
            content_docs.append("```\n")
    
    if file_count >= max_file_count:
        content_docs.append(f"⚠️ Only showing {max_file_count} files to avoid excessive output size.\n")
    
    return "\n".join(content_docs)

def calculate_project_stats(root_dir: Path, exclude_dirs: List[str] = None) -> Dict[str, Any]:
    """
    Calculate statistics about the project.
    
    Args:
        root_dir: Path to the project root
        exclude_dirs: List of directory names to exclude
        
    Returns:
        Dictionary of project statistics
    """
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    stats = {
        "total_files": 0,
        "total_directories": 0,
        "languages": Counter(),
        "file_extensions": Counter(),
        "total_lines_of_code": 0,
        "largest_files": [],
        "newest_files": []
    }
    
    # Queue of all files to process
    files_to_process = []
    
    # First, count directories and collect files
    for path in root_dir.glob("**/*"):
        # Skip excluded directories
        if any(excluded in str(path) for excluded in exclude_dirs):
            continue
            
        if path.is_dir():
            stats["total_directories"] += 1
        elif path.is_file():
            files_to_process.append(path)
            stats["total_files"] += 1
    
    # Process each file for detailed stats (limit to first 1000 for performance)
    for file_path in files_to_process[:1000]:
        # Update extension counter
        stats["file_extensions"][file_path.suffix] += 1
        
        # Update language counter
        language = get_file_language(file_path)
        stats["languages"][language] += 1
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Add to largest files (keep top 5)
        stats["largest_files"].append((str(file_path.relative_to(root_dir)), file_size))
        stats["largest_files"] = sorted(stats["largest_files"], key=lambda x: x[1], reverse=True)[:5]
        
        # Add to newest files (keep top 5)
        modification_time = file_path.stat().st_mtime
        stats["newest_files"].append((str(file_path.relative_to(root_dir)), modification_time))
        stats["newest_files"] = sorted(stats["newest_files"], key=lambda x: x[1], reverse=True)[:5]
        
        # Count lines of code (only for text files)
        if file_size < 1_000_000:  # Skip files larger than 1MB
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
                    stats["total_lines_of_code"] += line_count
            except:
                pass  # Skip files that can't be read
    
    # Convert newest files timestamps to readable dates
    stats["newest_files"] = [
        (path, datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        for path, timestamp in stats["newest_files"]
    ]
    
    # Format largest files sizes
    stats["largest_files"] = [
        (path, f"{size / 1024:.1f} KB" if size < 1_000_000 else f"{size / 1024 / 1024:.1f} MB")
        for path, size in stats["largest_files"]
    ]
    
    # Get top languages
    stats["top_languages"] = stats["languages"].most_common(5)
    
    return stats

def generate_project_documentation(
    root_dir: Path = None,
    output_path: str = None,
    project_name: str = None,
    generate_html: bool = True,
    max_file_count: int = 20
) -> Tuple[Path, Optional[Path]]:
    """
    Generate comprehensive project documentation.
    
    Args:
        root_dir: Path to the project root (auto-detected if None)
        output_path: Custom output path (default: PROJECT_DOCUMENTATION.md in root)
        project_name: Custom project name (default: inferred from directory)
        generate_html: Whether to generate HTML documentation as well
        max_file_count: Maximum number of files to include in content section
        
    Returns:
        Tuple of (markdown_path, html_path)
    """
    # Auto-detect project root if not provided
    if root_dir is None:
        root_dir = get_project_root()
    
    root_dir = Path(root_dir)
    
    # Infer project name if not provided
    if project_name is None:
        # Try to get from setup.py, pyproject.toml, or package.json
        if (root_dir / "setup.py").exists():
            try:
                setup_content = get_file_content(root_dir / "setup.py")
                name_match = re.search(r'name=[\'"]([^\'"]+)[\'"]', setup_content)
                if name_match:
                    project_name = name_match.group(1)
            except:
                pass
        
        # Check pyproject.toml
        if project_name is None and (root_dir / "pyproject.toml").exists():
            try:
                toml_content = get_file_content(root_dir / "pyproject.toml")
                name_match = re.search(r'name\s*=\s*[\'"]([^\'"]+)[\'"]', toml_content)
                if name_match:
                    project_name = name_match.group(1)
            except:
                pass
                
        # Use directory name as fallback
        if project_name is None:
            project_name = root_dir.name.replace("_", " ").replace("-", " ").title()
    
    # Set default output path if not provided
    if output_path is None:
        output_path = root_dir / "PROJECT_DOCUMENTATION.md"
    else:
        output_path = Path(output_path)
    
    # Create documentation header
    doc = [
        f"# {project_name} - Project Documentation",
        f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Project Overview",
        f"{project_name} is organized as follows. This documentation provides a comprehensive overview of the project structure, key modules, and implementation details.",
        "\n## Project Structure\n"
    ]
    
    # Add directory tree
    doc.append("```")
    doc.append(generate_directory_tree(root_dir))
    doc.append("```\n")
    
    # Add project statistics
    try:
        stats = calculate_project_stats(root_dir)
        doc.append("## Project Statistics\n")
        doc.append(f"- **Total Files:** {stats['total_files']}")
        doc.append(f"- **Total Directories:** {stats['total_directories']}")
        doc.append(f"- **Total Lines of Code:** {stats['total_lines_of_code']:,}")
        
        # Top languages
        doc.append("\n### Language Distribution\n")
        if stats['top_languages']:
            for language, count in stats['top_languages']:
                doc.append(f"- **{language.title()}:** {count} files")
        
        # Largest files
        doc.append("\n### Largest Files\n")
        if stats['largest_files']:
            for path, size in stats['largest_files']:
                doc.append(f"- `{path}`: {size}")
        
        # Newest files
        doc.append("\n### Recently Modified Files\n")
        if stats['newest_files']:
            for path, timestamp in stats['newest_files']:
                doc.append(f"- `{path}`: {timestamp}")
    except Exception as e:
        doc.append("Error calculating project statistics: " + str(e))
    
    # Add module documentation
    doc.append("\n## Module Documentation\n")
    doc.append(generate_module_documentation(root_dir))
    
    # Add file content documentation
    doc.append(generate_file_content_documentation(root_dir, max_file_count=max_file_count))
    
    # Add section on usage examples (if README contains examples, these will already be included)
    doc.append("\n## Getting Started\n")
    doc.append("To use this project, follow these steps:")
    doc.append("\n1. **Install dependencies**")
    
    # Check if requirements.txt exists
    if (root_dir / "requirements.txt").exists():
        doc.append("```bash")
        doc.append("pip install -r requirements.txt")
        doc.append("```\n")
    elif (root_dir / "pyproject.toml").exists():
        doc.append("```bash")
        doc.append("pip install .")
        doc.append("# Or for development:")
        doc.append("pip install -e .")
        doc.append("```\n")
    else:
        doc.append("Install the necessary dependencies for your project.\n")
    
    doc.append("2. **Explore the codebase**")
    doc.append("\n   See the module documentation and code samples above.\n")
    
    # Join all sections
    documentation = "\n".join(doc)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write documentation to markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"Markdown documentation generated at: {output_path}")
    
    html_output_path = None
    # Generate HTML version if requested
    if generate_html:
        html_output_path = output_path.with_suffix('.html')
        
        try:
            import pygments
            from pygments.formatters import HtmlFormatter
            has_pygments = True
        except ImportError:
            has_pygments = False
            print("pygments not installed. Code syntax highlighting in HTML will be basic.")
        
        # Create HTML with styling and syntax highlighting if available
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{project_name} - Project Documentation</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    margin-top: 24px;
                    margin-bottom: 16px;
                    font-weight: 600;
                    color: #0366d6;
                }}
                h1 {{ font-size: 2em; padding-bottom: 10px; border-bottom: 1px solid #eaecef; }}
                h2 {{ font-size: 1.5em; padding-bottom: 8px; border-bottom: 1px solid #eaecef; }}
                h3 {{ font-size: 1.25em; }}
                pre {{
                    padding: 16px;
                    overflow: auto;
                    font-size: 85%;
                    line-height: 1.45;
                    background-color: #f6f8fa;
                    border-radius: 6px;
                }}
                code {{
                    padding: 0.2em 0.4em;
                    margin: 0;
                    font-size: 85%;
                    background-color: rgba(27, 31, 35, 0.05);
                    border-radius: 3px;
                }}
                pre code {{
                    padding: 0;
                    background-color: transparent;
                }}
                blockquote {{
                    padding: 0 1em;
                    color: #6a737d;
                    border-left: 0.25em solid #dfe2e5;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 16px;
                }}
                table th, table td {{
                    padding: 6px 13px;
                    border: 1px solid #dfe2e5;
                }}
                table tr {{
                    background-color: #fff;
                    border-top: 1px solid #c6cbd1;
                }}
                table tr:nth-child(2n) {{
                    background-color: #f6f8fa;
                }}
                a {{
                    color: #0366d6;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                .timestamp {{
                    color: #6a737d;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div id="content">
                {markdown.markdown(documentation, extensions=['fenced_code', 'tables', 'codehilite'] if has_pygments else ['fenced_code', 'tables'])}
            </div>
            <footer>
                <p class="timestamp">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Created with Project Documentation Generator</p>
            </footer>
        </body>
        </html>
        """
        
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML documentation generated at: {html_output_path}")
    
    return output_path, html_output_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive project documentation with consistent content between formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--project-dir", "-p",
        type=str,
        default=None,
        help="Project name (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable HTML documentation generation"
    )
    
    parser.add_argument(
        "--max-files", "-m",
        type=int,
        default=20,
        help="Maximum number of files to include in the content section"
    )
    
    parser.add_argument(
        "--exclude-dirs", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Additional directories to exclude"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up the project root
    if args.project_dir:
        project_root = Path(args.project_dir)
    else:
        project_root = get_project_root()
        print(f"Auto-detected project root: {project_root}")
    
    # Set up exclude directories
    exclude_dirs = DEFAULT_EXCLUDE_DIRS
    if args.exclude_dirs:
        exclude_dirs.extend(args.exclude_dirs)
    
    # Generate the documentation
    print(f"Generating documentation for project.")
    md_path, html_path = generate_project_documentation(
        root_dir=project_root,
        generate_html=not args.no_html,
        max_file_count=args.max_files
    )
    
    print(f"Documentation generation complete!")
    print(f"Markdown: {md_path}")
    if html_path:
        print(f"HTML: {html_path}")
    
    # Check if we can open the file for the user
    try:
        if html_path and shutil.which('xdg-open'):  # Linux
            os.system(f'xdg-open "{html_path}"')
        elif html_path and shutil.which('open'):  # macOS
            os.system(f'open "{html_path}"')
        elif html_path and os.name == 'nt':  # Windows
            os.system(f'start "" "{html_path}"')
    except:
        pass  # No need to error if we can't open the file automatically

if __name__ == "__main__":
    main()