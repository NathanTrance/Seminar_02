import ast
import logging

logger = logging.getLogger(__name__)

SUSPICIOUS_IMPORTS = {
    "os", "subprocess", "shutil", "socket", "requests",
    "urllib", "urllib.request", "urllib.parse",
    "base64", "codecs", "ctypes", "pickle", "shelve",
    "marshal", "tempfile", "sys",
}

SUSPICIOUS_FUNCTIONS = {
    "os.system", "os.popen", "os.exec*", "os.execl", "os.execle",
    "os.execlp", "os.execv", "os.execve", "os.execvp",
    "os.fork", "os.spawn*", "subprocess.run", "subprocess.Popen",
    "subprocess.call", "subprocess.check_call",
    "eval", "exec", "compile",
    "requests.get", "requests.post", "urllib.request.urlopen",
    "urllib.request.urlretrieve",
    "open", "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree",
    "os.environ.get", "os.getenv",
    "base64.b64decode", "base64.b64encode",
    "ctypes.CDLL", "ctypes.WinDLL",
    "pickle.load", "pickle.loads", "marshal.load", "marshal.loads",
    "socket.connect", "socket.send", "socket.recv",
    "tempfile.mkstemp", "tempfile.mkdtemp",
    "os.chmod", "os.chown",
}


class ASTFeatureExtractor:
    def __init__(self, code: str):
        self.code = code
        self.tree: ast.AST | None = None
        self.parse_error: str | None = None
        self._parse()

    def _parse(self) -> None:
        if not self.code:
            self.parse_error = "empty code"
            return
        try:
            self.tree = ast.parse(self.code)
        except SyntaxError as e:
            self.parse_error = str(e)

    def extract_imports(self) -> list[str]:
        if self.tree is None:
            return []
        imports = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full = f"{module}.{alias.name}" if module else alias.name
                    imports.append(full)
        return imports

    def extract_calls(self) -> list[dict]:
        if self.tree is None:
            return []
        calls = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = node.func
                call_str = self._format_call(func)
                line = getattr(node, "lineno", 0)
                calls.append({"function": call_str, "line": line})
        return calls

    def _format_call(self, func: ast.AST) -> str:
        if isinstance(func, ast.Attribute):
            base = self._format_call(func.value)
            return f"{base}.{func.attr}" if base else func.attr
        elif isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Call):
            return self._format_call(func.func) + "(...)"
        return str(func)

    def detect_behaviors(self) -> dict[str, list[int]]:
        behaviors = {
            "shell_execution": [],
            "process_creation": [],
            "network_access": [],
            "file_write": [],
            "file_delete": [],
            "dynamic_execution": [],
            "environment_access": [],
            "base64_decode": [],
            "remote_download": [],
            "persistence": [],
            "obfuscation": [],
        }

        if self.tree is None:
            return behaviors

        imports = self.extract_imports()
        import_set = set(imports)
        calls = self.extract_calls()

        has_subprocess = "subprocess" in import_set
        has_os = "os" in import_set
        has_requests = "requests" in import_set
        has_urllib = any("urllib" in i for i in imports)
        has_socket = "socket" in import_set
        has_base64 = "base64" in import_set
        has_ctypes = "ctypes" in import_set
        has_shutil = "shutil" in import_set

        for call in calls:
            fn = call["function"]
            line = call["line"]

            if has_subprocess and any(
                x in fn for x in ["Popen", "run", "call", "check_call"]
            ):
                behaviors["process_creation"].append(line)
                behaviors["shell_execution"].append(line)

            if "os.system" in fn or "os.popen" in fn:
                behaviors["shell_execution"].append(line)

            if has_requests and "requests.get" in fn:
                behaviors["network_access"].append(line)
                behaviors["remote_download"].append(line)

            if has_urllib and any(
                x in fn for x in ["urlopen", "urlretrieve"]
            ):
                behaviors["network_access"].append(line)
                behaviors["remote_download"].append(line)

            if has_socket and any(x in fn for x in ["connect", "send"]):
                behaviors["network_access"].append(line)

            if "eval" == fn or "exec" == fn or "compile" == fn:
                behaviors["dynamic_execution"].append(line)

            if "open" == fn and has_shutil:
                behaviors["file_write"].append(line)

            if "os.remove" in fn or "os.unlink" in fn or "shutil.rmtree" in fn:
                behaviors["file_delete"].append(line)

            if "os.getenv" in fn or "os.environ.get" in fn:
                behaviors["environment_access"].append(line)

            if has_base64 and "base64.b64decode" in fn:
                behaviors["base64_decode"].append(line)

            if "os.chmod" in fn:
                behaviors["persistence"].append(line)

        return behaviors

    def to_behavior_text(self) -> str:
        imports = self.extract_imports()
        suspicious_imports_found = [i for i in imports if i in SUSPICIOUS_IMPORTS]
        behaviors = self.detect_behaviors()
        active = [k for k, v in behaviors.items() if v]
        calls = self.extract_calls()
        suspicious_calls = [
            c["function"] for c in calls
            if any(s in c["function"] for s in SUSPICIOUS_FUNCTIONS)
        ]

        parts = []
        if suspicious_imports_found:
            parts.append(f"imports={','.join(sorted(set(suspicious_imports_found)))}")
        if active:
            parts.append(f"behaviors={','.join(sorted(active))}")
        if suspicious_calls:
            parts.append(f"calls={','.join(sorted(set(suspicious_calls)))}")

        return "; ".join(parts) if parts else "benign_code"
