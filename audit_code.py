import ast
import os
import re

def find_stubs_and_hardcoded(root_dir):
    stubs = []
    hardcoded = []
    
    # Simple regex for hardcoded API keys or suspicious strings
    api_key_pattern = re.compile(r'(api_key|secret|password|token)\s*=\s*["\'][a-zA-Z0-9_\-]+["\']', re.IGNORECASE)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
                
            filepath = os.path.join(dirpath, filename)
            
            with open(filepath, 'r') as f:
                try:
                    content = f.read()
                    tree = ast.parse(content)
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
                    continue

                # Check for bad hardcoded values
                for i, line in enumerate(content.splitlines()):
                    if api_key_pattern.search(line):
                        # Filter out os.getenv or settings usage if regex was too aggressive?
                        # Actually regex is looking for literal string assignment.
                        # Exclude assignments that look like placeholders or reading from env (though AST is better for that)
                        if "os.getenv" not in line and "settings." not in line:
                             hardcoded.append((filepath, i+1, line.strip()))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for empty/stub functions
                        if not node.body:
                            stubs.append((filepath, node.lineno, node.name, "Empty body"))
                            continue
                        
                        if len(node.body) == 1:
                            stmt = node.body[0]
                            if isinstance(stmt, ast.Pass):
                                stubs.append((filepath, node.lineno, node.name, "Only pass"))
                            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
                                stubs.append((filepath, node.lineno, node.name, "Only ..."))
                            elif isinstance(stmt, ast.Return) and stmt.value is None:
                                # return None is valid sometimes, but worth checking
                                # stubs.append((filepath, node.lineno, node.name, "Only return None"))
                                pass
                            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                                # checking for explicit return None again
                                pass
                                
    return stubs, hardcoded

stubs, hardcoded = find_stubs_and_hardcoded('src')

print("=== STUBS FOUND ===")
for s in stubs:
    print(f"{s[0]}:{s[1]} - {s[2]} ({s[3]})")

print("\n=== HARDCODED VALUES FOUND ===")
for h in hardcoded:
    print(f"{h[0]}:{h[1]} - {h[2]}")
