#!/usr/bin/env python3
"""Flag non-raw Python string literals whose backslashes look like TeX eaten by Python escapes
(\\times -> tab+'imes', \\frac -> formfeed, \\beta -> backspace, \\nabla -> newline...)."""
import io, re, sys, tokenize
BAD = re.compile(r'(?<!\\)\\(?:[tfbavr][a-zA-Z]|n(?:abla|eq|u\b|ot|i\b|ewline|orm|atural|eg))')
rc = 0
for path in sys.argv[1:]:
    for tok in tokenize.generate_tokens(io.StringIO(open(path).read()).readline):
        if tok.type != tokenize.STRING:
            continue
        prefix = re.match(r'[a-zA-Z]*', tok.string).group(0).lower()
        if 'r' in prefix:
            continue
        for m in BAD.finditer(tok.string):
            line = tok.start[0] + tok.string[:m.start()].count('\n')
            print(f"{path}:{line}: {tok.string[max(0,m.start()-20):m.end()+10]!r}")
            rc = 1
sys.exit(rc)
