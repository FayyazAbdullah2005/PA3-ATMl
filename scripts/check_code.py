"""Check compilation and actual module-global references, including star imports."""
from pathlib import Path
import builtins
import importlib
import py_compile
import symtable
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/".testdeps").exists():
    sys.path.insert(0,str(ROOT/".testdeps"))


def main():
    errors=[]
    paths=sorted((ROOT/"pa3").glob("*.py"))
    for path in paths:
        py_compile.compile(str(path),doraise=True)
        module=importlib.import_module("pa3."+path.stem)
        tree=symtable.symtable(path.read_text(encoding="utf-8"),str(path),"exec")
        known=set(vars(module))|set(vars(builtins))
        def visit(table):
            for symbol in table.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in known:
                    errors.append(f"{path.name}:{table.get_lineno()} {table.get_name()}: undefined global {symbol.get_name()}")
            for child in table.get_children():
                visit(child)
        visit(tree)
    py_compile.compile(str(ROOT/"modal_app.py"),doraise=True)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Compiled {len(paths)} experiment modules; no unresolved module-global references.")


if __name__=="__main__":
    main()

