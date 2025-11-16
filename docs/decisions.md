# Key Project Decisions & Justifications

## 1. Feature Selection Decision
**Decision**: Excluded all cognitive test scores and clinical measurements
**Justification**: Competition rules prohibit medical features that require doctor's interpretation

## 2. Borderline Features Handling
**Decision**: Included patient-known conditions (Hypertension, Depression)
**Justification**: These are conditions that patients are typically aware of through diagnosis

## 3. Model Selection Strategy
**Decision**: Start with multiple baseline models before optimization
**Justification**: Ensures we compare different approaches and don't over-optimize early

## 4. Evaluation Metric Focus
**Decision**: Prioritize Recall alongside Accuracy
**Justification**: In medical risk prediction, identifying true positives (actual dementia cases) is crucial