# UiO-66 Review Explorer

A single-file Streamlit app built from the uploaded review manuscript:

**Perspective of UiO-66-Derived Metal-Organic Frameworks for CO2 Capture, Separation, and Conversion**

The placeholder CSVs were replaced with cleaned datasets extracted from the review tables, plus two small narrative datasets built from the LCA and outlook sections.

## What is included

- `app.py`  
  Single-file Streamlit application. It includes the optional OpenAI assistant directly in the same file, so no separate `ai_module.py` is needed.
- `requirements.txt`
- `data/`
  - `industrial_deployments.csv`
  - `benchmark_gravimetric_uptake.csv`
  - `benchmark_volumetric_uptake.csv`
  - `mixed_gas_benchmarks.csv`
  - `uio66_capture_design_space.csv`
  - `pure_membranes.csv`
  - `mixed_matrix_membranes.csv`
  - `co2_reduction_redox_potentials.csv`
  - `catalytic_conversion.csv`
  - `photocatalytic_conversion.csv`
  - `lca_hotspots.csv`
  - `lca_case_studies.csv`
  - `outlook_priorities.csv`

## App structure

The app is organised into these sections:

1. **Home**  
   Scope, industrial examples, and redox-potential context.

2. **Benchmarks**  
   - Gravimetric uptake at 298 K and 1 bar  
   - Volumetric uptake at 298 K  
   - Mixed-gas benchmark comparisons

3. **UiO-66 design space**  
   Interactive exploration of the large UiO-66 adsorption table, grouped by strategy:
   - Pure UiO-66
   - Direct functionalisation
   - Post-synthetic functionalisation
   - Post-synthetic metalation
   - Post-synthetic exchange
   - Hydrophobic UiO-66
   - Pore size control
   - Defect engineering
   - Porous liquids
   - Composites

4. **Membranes**  
   Separate views for pure UiO-66-based membranes and UiO-66-filled MMMs.

5. **Catalysis**

6. **Photocatalysis**

7. **Sustainability**  
   Uses the paper's LCA hotspots table plus narrative case studies and outlook priorities.

8. **Data browser**

9. **Ask the review**  
   Optional OpenAI-backed assistant using only the loaded dataset context.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional OpenAI setup

The AI tab is optional. To enable it, set one of these environment variables before launching Streamlit:

```bash
export OPENAI_API_KEY="your_key_here"
```

The app also checks `openai_api_key` and `openai_api_key2`, and it can use Streamlit secrets if you prefer that route.

## Notes on the data

- The original uploaded CSV files were placeholder examples and were not suitable for the paper in its current form.
- The new CSV files were rebuilt from the paper tables in the uploaded DOCX.
- Two extra CSV files were added:
  - `lca_case_studies.csv`
  - `outlook_priorities.csv`

These are structured from the review's narrative sections so the sustainability tab can show more than the single hotspot table.

## Suggested next refinement

If you want to push the app further later, the next good upgrade would be to add the paper figures as annotated image panels or redraw a few of the key schematics directly in the interface.
