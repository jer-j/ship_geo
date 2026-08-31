---
title: Background
---

The design separates form parameters from representation coefficients. Form
parameters are quantities a naval architect reasons about, such as section
area, waterline breadth, tangent angles, and centroids. B-spline coefficients
are internal states determined by a fairness solve.

This separation follows the early fairness-based hull parameterization work of
Harries. It also fits the LSDO computational model: form parameters are CSDL
inputs, spline coefficients are implicit states, and evaluated geometry is a
differentiable output for downstream analysis.

## Bibliography

```{bibliography} references.bib
```
