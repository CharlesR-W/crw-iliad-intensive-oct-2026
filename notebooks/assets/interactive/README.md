# Interactive figures (MDL day)

Each figure is a standalone HTML page, available to explore directly from its notebook.
Verification scripts are in `checks/`.

| File | Workbook (spec v2) | What it shows |
|---|---|---|
| `spectral_filters.html` | WB2 §4, WB5 §1 | ridge and early-stopping filters (half-points match at t = ln2/λ); scaling exponent set by the data spectrum |
| `double_descent.html` | WB2 §5 | excess risk vs p/n: the peak is variance, and optimal ridge removes it |
| `sgd_stationary_vs_posterior.html` | WB2 §8 | SGD's stationary spread (∝ H^(γ−1)) vs the posterior's (∝ H⁻¹), with a γ slider; deep-linkable with `?gamma=&kappa=&eta=&B=&phi=` |
| `superposition.html` | WB6 §5 | phase diagram for 2 features in 1 dimension, and polygons forming in the browser; generated from `checks/superposition_template.html`, so don't edit the output HTML |
