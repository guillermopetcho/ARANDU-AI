import sys
import ast

with open('engine/controller.py', 'r') as f:
    code = f.read()

tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'is_healthy_reorg':
                print(f"Found is_healthy_reorg assignment at line {node.lineno}")
                print(ast.unparse(node))
