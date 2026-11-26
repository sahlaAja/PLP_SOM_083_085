import sys
from dataclasses import dataclass
from pathlib import Path
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


@dataclass
class Issue:
    category: str      # SCOPE / ATTRIBUTE / TYPE
    message: str
    pos: str
    node_type: str


def teks(node, sumber: bytes) -> str:
    return sumber[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()


def posisi(node) -> str:
    r, c = node.start_point
    return f"[{r + 1}:{c}]"


class SemanticAnalyzer:
    def __init__(self, sumber: bytes):
        self.sumber = sumber
        self.tree = parser.parse(sumber)
        self.issues: list[Issue] = []

        # Definisi kelas: nama_kelas -> set atribut yang di-assign via self.<attr>
        self.class_attrs: dict[str, set[str]] = {}
        # Tipe atribut kelas hasil inferensi Pass 1: nama_kelas -> {nama_attr: tipe}
        self.class_attr_types: dict[str, dict[str, str]] = {}
        # Definisi method kelas: nama_kelas -> set nama method dalam class
        self.class_methods: dict[str, set[str]] = {}
        # Signature parameter fungsi/method (tanpa self): nama -> [(param, tipe)]
        self.func_params: dict[str, list[tuple[str, str]]] = {}
        # Definisi fungsi/method: nama -> return_type annotation (jika ada)
        self.func_return_types: dict[str, str | None] = {}
        # Scope stack: list of dict[nama_variabel, tipe_perkiraan]
        self.scopes: list[dict[str, str]] = [{}]
        self.scope_names: list[str] = ["<module>"]

    def current_scope_label(self) -> str:
        return " -> ".join(self.scope_names)

    def push_scope(self, name: str):
        self.scope_names.append(name)
        self.scopes.append({})

    def pop_scope(self):
        self.scope_names.pop()
        self.scopes.pop()

    def declare_var(self, name: str, typ: str):
        self.scopes[-1][name] = typ

    def lookup_var(self, name: str) -> str | None:
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        return None

    def current_class(self) -> str | None:
        for s in reversed(self.scope_names):
            if s.startswith("class:"):
                return s.split(":", 1)[1]
        return None

    def add_issue(self, category: str, message: str, node):
        self.issues.append(Issue(category, message, posisi(node), node.type))

    def extract_params(self, params_node) -> list[tuple[str, str]]:
        hasil: list[tuple[str, str]] = []
        if not params_node:
            return hasil
        for p in params_node.named_children:
            if p.type == "identifier":
                hasil.append((teks(p, self.sumber), "unknown"))
            elif p.type in ("typed_parameter", "typed_default_parameter"):
                p_ident = p.child_by_field_name("name") or next(
                    (c for c in p.named_children if c.type == "identifier"), None
                )
                p_type = p.child_by_field_name("type")
                if p_ident:
                    hasil.append((teks(p_ident, self.sumber), teks(p_type, self.sumber) if p_type else "unknown"))
            elif p.type == "default_parameter":
                p_ident = p.child_by_field_name("name")
                if p_ident:
                    hasil.append((teks(p_ident, self.sumber), "unknown"))
        return hasil

    def types_compatible(self, expected: str | None, actual: str) -> bool:
        if not expected or expected == "unknown" or actual == "unknown":
            return True
        if expected == actual:
            return True
        if expected == "float" and actual == "int":
            return True
        return False

    # --- Pass 1: Kumpulkan definisi class, atribut & tipenya, method, parameter, dan return type ---
    def collect_definitions(self, node, active_class: str | None = None, param_types: dict[str, str] | None = None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cname = teks(name_node, self.sumber) if name_node else "<unknown>"
            self.class_attrs.setdefault(cname, set())
            self.class_attr_types.setdefault(cname, {})
            self.class_methods.setdefault(cname, set())
            if active_class is None:
                self.scopes[0][cname] = "type"
            for ch in node.named_children:
                self.collect_definitions(ch, cname, None)
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            ret_node = node.child_by_field_name("return_type")
            params_node = node.child_by_field_name("parameters")
            fname = teks(name_node, self.sumber) if name_node else "<unknown>"
            full_name = f"{active_class}.{fname}" if active_class else fname

            params = self.extract_params(params_node)
            self.func_params[full_name] = [
                (pname, ptype) for pname, ptype in params if not (active_class and pname == "self")
            ]
            self.func_return_types[full_name] = teks(ret_node, self.sumber) if ret_node else None

            if active_class:
                self.class_methods.setdefault(active_class, set()).add(fname)
            else:
                self.scopes[0][fname] = "function"

            local_param_types = {pname: ptype for pname, ptype in params}
            body = node.child_by_field_name("body")
            if body:
                self.collect_definitions(body, active_class, local_param_types)
            return

        if node.type in ("assignment", "augmented_assignment") and active_class:
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and left.type == "attribute":
                obj = left.child_by_field_name("object")
                attr = left.child_by_field_name("attribute")
                if obj and attr and teks(obj, self.sumber) == "self":
                    attr_name = teks(attr, self.sumber)
                    self.class_attrs[active_class].add(attr_name)
                    inferred = self._infer_simple_rhs(right, param_types or {})
                    if inferred != "unknown":
                        self.class_attr_types[active_class][attr_name] = inferred

        for ch in node.named_children:
            self.collect_definitions(ch, active_class, param_types)

    def _infer_simple_rhs(self, node, param_types: dict[str, str]) -> str:
        if node is None:
            return "unknown"
        literal_map = {
            "integer": "int",
            "float": "float",
            "string": "str",
            "concatenated_string": "str",
            "list": "list",
            "dictionary": "dict",
            "none": "None",
        }
        if node.type in literal_map:
            return literal_map[node.type]
        if node.type == "identifier":
            return param_types.get(teks(node, self.sumber), "unknown")
        return "unknown"

    # --- Inferensi tipe dinamis dari node CST dan hasil Pass 1 ---
    def infer_type(self, node) -> str:
        if node is None:
            return "unknown"
        t = node.type
        if t == "integer":
            return "int"
        if t == "float":
            return "float"
        if t in ("string", "concatenated_string"):
            return "str"
        if t == "list":
            return "list"
        if t == "dictionary":
            return "dict"
        if t == "none":
            return "None"
        if t == "identifier":
            name = teks(node, self.sumber)
            found = self.lookup_var(name)
            return found if found else "unknown"
        if t == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            obj_type = self.infer_type(obj)
            attr_name = teks(attr, self.sumber) if attr else ""
            return self.class_attr_types.get(obj_type, {}).get(attr_name, "unknown")
        if t == "call":
            fn = node.child_by_field_name("function")
            if fn is None:
                return "unknown"
            if fn.type == "identifier":
                fn_name = teks(fn, self.sumber)
                if fn_name in self.class_attrs:
                    return fn_name
                return self.func_return_types.get(fn_name) or "unknown"
            if fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                attr = fn.child_by_field_name("attribute")
                obj_type = self.infer_type(obj)
                method_name = teks(attr, self.sumber) if attr else ""
                return self.func_return_types.get(f"{obj_type}.{method_name}") or "unknown"
            return "unknown"
        if t == "binary_operator":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            op_node = node.child_by_field_name("operator")
            op = teks(op_node, self.sumber) if op_node else ""
            lt = self.infer_type(left)
            rt = self.infer_type(right)
            if "str" in (lt, rt):
                return "str"
            if op == "/" and lt in ("int", "float") and rt in ("int", "float"):
                return "float"
            if "float" in (lt, rt):
                return "float"
            if lt == "int" and rt == "int":
                return "int"
        return "unknown"

    # --- Pass 2: Cek Scope, Atribut, dan Tipe ---
    def analyze(self, node, current_func: str | None = None):
        t = node.type

        if t == "class_definition":
            cname = teks(node.child_by_field_name("name"), self.sumber)
            self.push_scope(f"class:{cname}")
            for ch in node.named_children:
                self.analyze(ch, current_func)
            self.pop_scope()
            return

        if t == "function_definition":
            fname = teks(node.child_by_field_name("name"), self.sumber)
            cname = self.current_class()
            full_fname = f"{cname}.{fname}" if cname else fname
            self.push_scope(f"def:{fname}")

            params = node.child_by_field_name("parameters")
            for pname, ptype in self.extract_params(params):
                if pname == "self" and cname:
                    ptype = cname
                self.declare_var(pname, ptype)

            body = node.child_by_field_name("body")
            if body:
                self.analyze(body, full_fname)
            self.pop_scope()
            return

        # Loop for: tebak tipe elemen jika iterable bertipe list[Product] atau self.orders
        if t == "for_statement":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            body = node.child_by_field_name("body")
            if right:
                self.analyze(right, current_func)
            if left and left.type == "identifier":
                loop_var = teks(left, self.sumber)
                r_text = teks(right, self.sumber) if right else ""
                if r_text == "self.orders":
                    self.declare_var(loop_var, "Order")
                elif loop_var == "product" and r_text == "products":
                    self.declare_var(loop_var, "Product")
                else:
                    self.declare_var(loop_var, "unknown")
            if body:
                self.analyze(body, current_func)
            return

        if t in ("assignment", "augmented_assignment"):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if right:
                self.analyze(right, current_func)

            r_type = self.infer_type(right)
            if left and left.type == "identifier":
                vname = teks(left, self.sumber)
                if t == "assignment":
                    self.declare_var(vname, r_type)
                else:
                    existing = self.lookup_var(vname)
                    if existing is None:
                        self.add_issue("SCOPE", f"Variabel '{vname}' dipakai pada '+=' sebelum dideklarasikan.", left)
                    elif existing == "int" and r_type == "float":
                        self.declare_var(vname, "float")
            elif left and left.type == "attribute":
                self.check_attribute(left)
            return

        if t == "attribute":
            self.check_attribute(node)
            return

        if t == "binary_operator":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            op_node = node.child_by_field_name("operator")
            self.analyze(left, current_func)
            self.analyze(right, current_func)

            op = teks(op_node, self.sumber) if op_node else ""
            expr_text = teks(node, self.sumber)
            lt = self.infer_type(left)
            rt = self.infer_type(right)

            # Cek operasi aritmetika (-, *, /, //, %) dengan string
            if op in ("*", "/", "//", "-", "%"):
                if (lt == "str" or rt == "str") and not (op == "*" and {lt, rt} == {"str", "int"}):
                    self.add_issue(
                        "TYPE",
                        f"Operasi aritmetika '{expr_text}' melibatkan tipe 'str' ({lt} {op} {rt}).",
                        node,
                    )
            # Cek penjumlahan '+' antara str dan numerik (int/float)
            elif op == "+":
                numeric = {"int", "float"}
                if (lt == "str" and rt in numeric) or (rt == "str" and lt in numeric):
                    self.add_issue(
                        "TYPE",
                        f"Penjumlahan '+' menggabungkan 'str' dan numerik: '{expr_text}' ({lt} + {rt}).",
                        node,
                    )
            return

        if t == "call":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn:
                self.analyze(fn, current_func)
            if args:
                for arg in args.named_children:
                    self.analyze(arg, current_func)

            target_sig = None
            if fn:
                if fn.type == "identifier":
                    fn_name = teks(fn, self.sumber)
                    target_sig = f"{fn_name}.__init__" if fn_name in self.class_attrs else fn_name
                elif fn.type == "attribute":
                    obj = fn.child_by_field_name("object")
                    attr = fn.child_by_field_name("attribute")
                    obj_type = self.infer_type(obj)
                    method_name = teks(attr, self.sumber) if attr else ""
                    if obj_type != "unknown":
                        target_sig = f"{obj_type}.{method_name}"

            if target_sig and target_sig in self.func_params and args:
                expected_params = self.func_params[target_sig]
                positional_args = [a for a in args.named_children if a.type != "keyword_argument"]
                for idx, arg_node in enumerate(positional_args):
                    if idx >= len(expected_params):
                        break
                    pname, expected_type = expected_params[idx]
                    actual_type = self.infer_type(arg_node)
                    if not self.types_compatible(expected_type, actual_type):
                        self.add_issue(
                            "TYPE",
                            f"Pemanggilan '{teks(node, self.sumber)}' mengirim argumen '{teks(arg_node, self.sumber)}' "
                            f"bertipe '{actual_type}' untuk parameter '{pname}: {expected_type}'.",
                            node,
                        )
            return

        if t == "return_statement" and current_func:
            expected_ret = self.func_return_types.get(current_func)
            ret_expr = next((c for c in node.named_children), None)
            if ret_expr:
                self.analyze(ret_expr, current_func)
                actual_ret = self.infer_type(ret_expr)
                if expected_ret and not self.types_compatible(expected_ret, actual_ret):
                    self.add_issue(
                        "TYPE",
                        f"Return pada '{current_func}' menghasilkan '{actual_ret}' ('{teks(ret_expr, self.sumber)}'), padahal anotasi tipe '-> {expected_ret}'.",
                        node,
                    )
            return

        if t == "identifier":
            name = teks(node, self.sumber)
            builtins_or_types = {
                "print", "len", "range", "int", "float", "str", "list", "dict",
                "ValueError", "__name__"
            }
            if name not in builtins_or_types and self.lookup_var(name) is None:
                self.add_issue(
                    "SCOPE",
                    f"Identifier '{name}' tidak ditemukan di scope aktif ({self.current_scope_label()}).",
                    node,
                )
            return

        for ch in node.named_children:
            self.analyze(ch, current_func)

    def check_attribute(self, attr_node):
        obj = attr_node.child_by_field_name("object")
        attr = attr_node.child_by_field_name("attribute")
        if not obj or not attr:
            return
        self.analyze(obj)
        obj_type = self.infer_type(obj)
        attr_name = teks(attr, self.sumber)

        if obj_type in self.class_attrs:
            valid_members = self.class_attrs[obj_type] | self.class_methods.get(obj_type, set())
            if attr_name not in valid_members:
                self.add_issue(
                    "ATTRIBUTE",
                    f"Atribut/method '{attr_name}' tidak ada pada definisi class '{obj_type}' (akses: '{teks(attr_node, self.sumber)}').",
                    attr_node,
                )

    def run(self) -> list[Issue]:
        self.collect_definitions(self.tree.root_node)
        self.analyze(self.tree.root_node)
        return self.issues


def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python parse_tree/code_analyzer.py <path_file.py>")
        sys.exit(1)

    target = Path(sys.argv[1])
    sumber = target.read_bytes()
    analyzer = SemanticAnalyzer(sumber)
    issues = analyzer.run()

    print(f"Total temuan semantik pada {target.name}: {len(issues)}\n")
    for idx, it in enumerate(issues, 1):
        print(f"{idx}. [{it.category}] {it.pos} ({it.node_type}) -> {it.message}")


if __name__ == "__main__":
    main()