import ast
import re
from typing import List, Tuple


class CodeParser:
    """
    A universal code parser that extracts function and class definitions
    from 20+ programming languages and frameworks.
    """

    EXTENSION_MAP = {
        # Programming languages
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "react",
        ".tsx": "react",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".swift": "swift",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".rs": "rust",
        ".kt": "kotlin",
        ".dart": "dart",
        ".scala": "scala",
        ".lua": "lua",
        ".r": "r",
        ".pl": "perl",
        ".sh": "shell",
        ".m": "matlab",
        # Framework-specific files
        ".vue": "vue",
        ".svelte": "svelte",
    }

    @staticmethod
    def detect_language(file_extension: str) -> str:
        """Detect programming language based on file extension."""
        return CodeParser.EXTENSION_MAP.get(file_extension.lower(), "unknown")

    @staticmethod
    def parse_code(code: str, file_extension: str) -> Tuple[List[str], List[str]]:
        """
        Parse the given code and extract function and class definitions.

        Args:
            code (str): The source code.
            file_extension (str): The file extension (e.g., '.py', '.vue').

        Returns:
            Tuple[List[str], List[str]]: (List of function names, List of class names)
        """
        language = CodeParser.detect_language(file_extension)

        if language == "python":
            return CodeParser._parse_python(code)
        elif language in {
            "java",
            "javascript",
            "typescript",
            "react",
            "csharp",
            "cpp",
            "c",
            "swift",
            "go",
            "ruby",
            "php",
            "rust",
            "kotlin",
            "dart",
            "scala",
            "lua",
            "r",
            "perl",
            "shell",
            "matlab",
            "vue",
            "svelte",
        }:
            return CodeParser._parse_general(code, language)
        else:
            return [], []

    @staticmethod
    def _parse_python(code: str) -> Tuple[List[str], List[str]]:
        """Extract function and class names from Python code using AST."""
        try:
            tree = ast.parse(code)
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            classes = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            return functions, classes
        except Exception:
            return [], []

    @staticmethod
    def _parse_general(code: str, language: str) -> Tuple[List[str], List[str]]:
        """
        Extract function and class names from various languages and frameworks
        using regex.
        """

        # Generic patterns
        class_pattern = r"\bclass\s+(\w+)"

        function_patterns = {
            # Regular programming languages
            "java": r"\b(?:public|private|protected|static|\s)*\s+\w+\s+(\w+)\s*\(",
            "javascript": r"\bfunction\s+(\w+)\s*\(|\b(\w+)\s*=\s*\(.*?\)\s*=>",
            "typescript": r"\bfunction\s+(\w+)\s*\(|\b(\w+)\s*=\s*\(.*?\)\s*=>",
            "react": r"\bfunction\s+(\w+)\s*\(|\bconst\s+(\w+)\s*=\s*\(.*?\)\s*=>",
            "csharp": r"\b(?:public|private|protected|static|\s)*\s+\w+\s+(\w+)\s*\(",
            "cpp": r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{",
            "c": r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{",
            "swift": r"\bfunc\s+(\w+)\s*\(",
            "go": r"\bfunc\s+(\w+)\s*\(",
            "ruby": r"\bdef\s+(\w+)",
            "php": r"\bfunction\s+(\w+)\s*\(",
            "rust": r"\bfn\s+(\w+)\s*\(",
            "kotlin": r"\bfun\s+(\w+)\s*\(",
            "dart": r"\b(?:static\s+)?\w+\s+(\w+)\s*\(",
            "scala": r"\bdef\s+(\w+)\s*\(",
            "lua": r"\bfunction\s+(\w+)\s*\(",
            "r": r"\b(\w+)\s*<- function\(",
            "perl": r"\bsub\s+(\w+)\s*\{",
            "shell": r"\b(\w+)\s*\(\)\s*\{",
            "matlab": r"\bfunction\s+[^\s]+\s+(\w+)\s*\(",
            # Frameworks
            "vue": r"\bexport\s+default\s+{\s*methods\s*:\s*{([^}]*)}|methods\s*:\s*{([^}]*)}",  # noqa: E501
            "svelte": r"\bexport\s+function\s+(\w+)\s*\(|\bfunction\s+(\w+)\s*\(",
        }

        function_pattern = function_patterns.get(language, None)
        functions = re.findall(function_pattern, code) if function_pattern else []
        functions = [f for f in functions if f]

        classes = re.findall(class_pattern, code)

        return functions, classes
