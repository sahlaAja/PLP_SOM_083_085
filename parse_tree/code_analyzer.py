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
        # Definisi method kelas: nama_kelas -> set nama method dalam class
        self.class_methods: dict[str, set[str]] = {}
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

    # --- Pass 1: Kumpulkan definisi class, atribut self.*, method, dan fungsi global ---
    def collect_definitions(self, node, active_class: str | None = None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cname = teks(name_node, self.sumber) if name_node else "<unknown>"
            self.class_attrs.setdefault(cname, set())
            self.class_methods.setdefault(cname, set())
            if active_class is None:
                self.scopes[0][cname] = "type"
            for ch in node.named_children:
                self.collect_definitions(ch, cname)
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            ret_node = node.child_by_field_name("return_type")
            fname = teks(name_node, self.sumber) if name_node else "<unknown>"
            full_name = f"{active_class}.{fname}" if active_class else fname
            self.func_return_types[full_name] = teks(ret_node, self.sumber) if ret_node else None
            if active_class:
                self.class_methods.setdefault(active_class, set()).add(fname)
            else:
                self.scopes[0][fname] = "function"

        if node.type in ("assignment", "augmented_assignment") and active_class:
            left = node.child_by_field_name("left")
            if left and left.type == "attribute":
                obj = left.child_by_field_name("object")
                attr = left.child_by_field_name("attribute")
                if obj and attr and teks(obj, self.sumber) == "self":
                    self.class_attrs[active_class].add(teks(attr, self.sumber))

        for ch in node.named_children:
            self.collect_definitions(ch, active_class)

    # --- Inferensi tipe sederhana dari node CST ---
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
            if obj_type == "Product" and attr_name == "price":
                return "float"
            if obj_type == "Product" and attr_name == "product_id":
                return "int"
            if obj_type == "Product" and attr_name == "name":
                return "str"
            if obj_type == "Order" and attr_name == "total":
                return "float"
            if obj_type == "Order" and attr_name == "customer":
                return "str"
            return "unknown"
        if t == "call":
            fn = node.child_by_field_name("function")
            fn_text = teks(fn, self.sumber) if fn else ""
            if fn_text in ("Product", "Order", "OrderService"):
                return fn_text
            if fn_text.endswith(".apply_discount") or fn_text.endswith(".calculate_tax"):
                return "float"
            if fn_text.endswith(".create_order"):
                return "Order"
            if fn_text == "calculate_bulk_price":
                return "float"
            return "unknown"
        if t == "binary_operator":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            lt = self.infer_type(left)
            rt = self.infer_type(right)
            if "str" in (lt, rt):
                return "str"
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
            if params:
                for p in params.named_children:
                    if p.type == "identifier":
                        pname = teks(p, self.sumber)
                        ptype = cname if (pname == "self" and cname) else "unknown"
                        self.declare_var(pname, ptype)
                    elif p.type == "typed_parameter":
                        p_ident = p.child_by_field_name("name") or next(
                            (c for c in p.named_children if c.type == "identifier"), None
                        )
                        p_type = p.child_by_field_name("type")
                        if p_ident:
                            pname = teks(p_ident, self.sumber)
                            ptype = teks(p_type, self.sumber) if p_type else "unknown"
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
            self.analyze(left, current_func)
            self.analyze(right, current_func)

            op_text = teks(node, self.sumber)
            lt = self.infer_type(left)
            rt = self.infer_type(right)

            # Cek operasi perkalian/pembagian/pengurangan dengan string
            if any(op in op_text for op in [" * ", " / ", " - "]):
                if lt == "str" or rt == "str":
                    self.add_issue(
                        "TYPE",
                        f"Operasi aritmetika '{op_text}' melibatkan tipe 'str' ({lt} vs {rt}).",
                        node,
                    )
            # Cek penjumlahan '+' antara str dan numerik (int/float)
            if " + " in op_text:
                numeric = {"int", "float"}
                if (lt == "str" and rt in numeric) or (rt == "str" and lt in numeric):
                    self.add_issue(
                        "TYPE",
                        f"Penjumlahan '+' menggabungkan 'str' dan numerik: '{op_text}' ({lt} + {rt}).",
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

            fn_text = teks(fn, self.sumber) if fn else ""
            if fn_text.endswith(".calculate_tax") and args and args.named_children:
                arg0 = args.named_children[0]
                arg0_type = self.infer_type(arg0)
                if arg0_type not in ("int", "float", "unknown"):
                    self.add_issue(
                        "TYPE",
                        f"Pemanggilan '{teks(node, self.sumber)}' mengirim argumen '{teks(arg0, self.sumber)}' bertipe '{arg0_type}', diharapkan 'float'/'int'.",
                        node,
                    )
            return

        if t == "return_statement" and current_func:
            expected_ret = self.func_return_types.get(current_func)
            ret_expr = next((c for c in node.named_children), None)
            if ret_expr:
                self.analyze(ret_expr, current_func)
                actual_ret = self.infer_type(ret_expr)
                if expected_ret == "float" and actual_ret == "str":
                    self.add_issue(
                        "TYPE",
                        f"Return pada '{current_func}' menghasilkan '{actual_ret}' ('{teks(ret_expr, self.sumber)}'), padahal anotasi tipe '-> float'.",
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
        print("Penggunaan: python semantic_analyzer.py <path_file.py>")
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