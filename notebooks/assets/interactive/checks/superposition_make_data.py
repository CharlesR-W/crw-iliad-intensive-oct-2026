"""Writes the compact phase table embedded in superposition.html (from superposition_check_out.json)."""
import json, numpy as np
d = json.load(open("superposition_check_out.json"))
LC = [[None if v is None else float(f"{v:.5e}") for v in row] for row in d["LC"]]
AC = [[None if v is None else [round(x, 4) for x in v] for v in row] for row in d["AC"]]
P = np.logspace(-2, 0, 81); RF = np.logspace(-1, 1, 81)
tr = []
for c in d["trained"]:
    j = int(np.argmin(abs(np.log(P) - np.log(c["p"])))); i = int(np.argmin(abs(np.log(RF) - np.log(c["r"]))))
    tr.append([j, i, c["got"], c["pred"]])
s = "const TABLE=" + json.dumps(dict(LC=LC, AC=AC, trained=tr), separators=(",", ":")) + ";"
open("superposition_data.js", "w").write(s); print(len(s))
tpl = open("superposition_template.html").read()
open("../superposition.html", "w").write(tpl.replace("/*DATA*/", s))
print("wrote ../superposition.html")
