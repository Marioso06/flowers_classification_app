---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  /* PowerPoint-like theme based on provided images */
  :root {
    --color-background: #ffffff;
    --color-foreground: #333333;
    --color-accent-yellow: #FFCC00;
    --color-accent-red: #FF5252;
    --font-header: 'Arial Black', 'Arial Bold', 'Arial', sans-serif;
    --font-body: 'Arial', sans-serif;
  }

  section {
    background-color: var(--color-background);
    color: var(--color-foreground);
    font-family: var(--font-body);
    padding: 50px 60px; /* Increased padding to reduce effective content area */
    background-image: 
      /* Yellow top bar */
      linear-gradient(90deg, var(--color-accent-yellow), var(--color-accent-yellow)),
      /* Red top bar (smaller) */
      linear-gradient(90deg, var(--color-accent-red), var(--color-accent-red));
    background-position: 
      top left,
      top right;
    background-size: 
      100% 10px,
      100% 5px;
    background-repeat: no-repeat;
    max-height: 100%; /* Limit height to 80% of original */
    max-width: 90%; /* Limit width to 80% of original */
    margin: 0 auto; /* Center the content */
  }

  /* Logo in top right */
  section::before {
    content: '';
    position: absolute;
    top: 15px;
    right: 20px;
    width: 45px;
    height: 45px;
    background-image: url('./resources/logo.png');
    background-repeat: no-repeat;
    background-size: contain;
  }

  /* Yellow curved element - removed as requested */

  /* Slide number */
  section::after {
    color: #888;
    font-size: 14px;
    right: 20px;
    bottom: 20px;
    content: attr(data-marpit-pagination);
    position: absolute;
  }

  h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-header);
    color: #333;
    margin-bottom: 0.5em;
  }

  h1 {
    font-size: 1.7em;
    font-weight: 900;
    margin-top: 35px;
  }

  h2 {
    font-size: 1.35em;
    font-weight: bold;
  }

  a {
    color: #0066cc;
    text-decoration: none;
  }

  strong {
    color: #333;
    font-weight: bold;
  }

  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    border: 1px solid #ddd;
  }

  th {
    background-color: #f2f2f2;
    color: #333;
    padding: 0.8em;
    border: 1px solid #ddd;
    font-weight: bold;
  }

  td {
    border: 1px solid #ddd;
    padding: 0.8em;
  }

  tr:nth-child(even) {
    background-color: #f9f9f9;
  }

  code {
    background-color: #f3f3f3;
    color: #333;
    padding: 0.2em 0.4em;
    border-radius: 3px;
  }

  ul li, ol li {
    margin-bottom: 0.54em;
    list-style-type: square;
    font-size: 0.85em;
  }

  /* Title slide */
  section.title {
    text-align: left;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  section.title h1 {
    font-size: 2.25em;
    width: 80%;
    color: #333;
  }

  section.title h2 {
    font-size: 1.35em;
    margin-top: 0.45em;
    width: 60%;
    color: #333;
  }

  /* Two-column layout */
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5em;
    max-width: 100%;
  }

  /* Footer styling */
  .footer {
    position: absolute;
    bottom: 20px;
    width: 90%;
    font-size: 0.8em;
    color: #888;
  }

  /* Override for feature tables */
  .feature-table {
    border: none;
  }
  
  .feature-table td {
    border: none;
    padding: 10px;
    vertical-align: top;
  }

  /* Custom bullet points */
  .custom-bullet {
    display: inline-block;
    width: 20px;
    height: 20px;
    background-color: var(--color-accent-yellow);
    color: #333;
    border-radius: 50%;
    text-align: center;
    margin-right: 7px;
    font-weight: bold;
    font-size: 0.8em;
    line-height: 20px;
  }
---

<!-- 
This presentation is formatted for Marp (https://marp.app/)
To view as slides:
1. Install Marp CLI or use the Marp for VS Code extension
2. Run: marp Explainable_AI_Slides.md -o Explainable_AI_Slides.pdf
-->

<!-- _class: title -->

# **Explainable Artificial Intelligence (XAI)**
## Understanding Model Decisions in the Era of Complex AI

<div style="position:absolute; bottom:80px; right:60px; width:300px;">
  <!-- External image removed -->
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

<!-- _class: default -->

# **The Problem: Black Box AI**

<div class="columns">
<div>

- Complex AI models (especially deep neural networks) often function as "black boxes"
- High accuracy but low transparency
- Difficult to understand **why** and **how** decisions are made
- Creates barriers to trust and adoption
- Particularly problematic in high-stakes domains (healthcare, finance, legal)

</div>
<div>

<!-- External image removed -->

</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

<!-- _class: default -->

# **What is Explainable AI?**

Explainable AI (XAI) encompasses techniques and methodologies that make AI systems **understandable to humans**

<!-- External image removed -->

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

<!-- _class: default -->

# **Key Concepts in XAI**

| Concept | Definition |
|---------|------------|
| **Explainability** | Ability to provide clear, meaningful justification for AI decisions in human-understandable terms |
| **Interpretability** | Ability to understand how a model works and which inputs influence outputs |
| **Transparency** | Openness about design, data, and decision-making processes |

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

<!-- _class: default -->

# **Explainability vs. Interpretability vs. Transparency**

<!-- External image removed -->

<div class="columns">
<div>

- **Explainability**: The "why" behind predictions
- **Interpretability**: The "how" of model operation

</div>
<div>

- **Transparency**: Visibility into the entire AI system
- **Critical distinctions for project implementation**

</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

<!-- _class: default -->

# **Why XAI Matters**

<div class="columns">
<div>

- **Responsibility & Accountability**
- **Building Trust & Confidence**
- **Model Adaptation & Improvement**
- **Regulatory Compliance**
- **Reducing Bias & Promoting Fairness**

</div>
<div>

- **Debugging & Error Detection**
- **Improved User Experience**
- **Knowledge Transfer**
- **Scientific Discovery**

<!-- External image removed -->

</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

# **Principles of Explainable AI**

<!-- External image removed -->

- **Transparency**: Comprehensible processes
- **Interpretability**: Understandable outputs
- **Explainability**: Clear reasoning
- **Justifiability**: Evidence-backed decisions
- **Robustness & Reliability**: Consistent performance
- **Causality**: Understanding cause-effect 
- **Fairness**: Mitigating biases

---

# **NIST Principles for XAI**

- **Meaningful**: Tailored to audience needs
- **Accurate**: Correctly reflecting system processes
- **Knowledge Limits**: Operating within designated boundaries
- **Confidence**: Indicating certainty levels

*Source: National Institute of Standards and Technology*

---

# **XAI Techniques: Overview**

<!-- External image removed -->

XAI techniques can be categorized as:
- **Model-specific** vs. **Model-agnostic**
- **Local** vs. **Global** explainability
- **Intrinsic** vs. **Post-hoc** methods

---

# **Model-Specific vs. Model-Agnostic Techniques**

<!-- External image removed -->

**Model-Specific Techniques:**
- Designed for particular model architectures
- Examples: Feature visualization for CNNs, attention maps for transformers

**Model-Agnostic Techniques:**
- Work with any machine learning model
- Examples: SHAP, LIME, permutation feature importance

---

# **Key Model-Specific Techniques**

- **Decision Trees & Rules**:
  - Inherently interpretable
  - Visual representation of decision paths

- **Linear/Logistic Regression**:
  - Coefficients indicate feature influence
  - Simple, transparent relationships

- **Attention Mechanisms**:
  - Highlight important input regions
  - Popular in NLP and computer vision

---

# **Key Model-Agnostic Techniques**

- **SHAP (SHapley Additive exPlanations)**:
  - Based on game theory
  - Assigns importance scores to features
  - Consistent, locally accurate explanations
  - Various visualization methods

- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Creates interpretable surrogate models
  - Focuses on individual predictions
  - Generates perturbed samples around target instance

---

# **SHAP Visualizations**

<!-- External image removed -->

- **Waterfall plots**: Feature contribution to specific prediction
- **Force plots**: Similar information in different format
- **Summary plots**: Overview of feature importance
- **Dependence plots**: Relationship between feature values and impact

---

# **More Model-Agnostic Techniques**

- **Partial Dependence Plots (PDP)**:
  - Show relationship between features and predictions
  - Marginal effect of one or two features

- **Permutation Feature Importance**:
  - Measures impact of shuffling feature values
  - Shows which features are most critical

- **Counterfactual Explanations**:
  - "What-if" scenarios
  - Shows changes needed to get different outcome

---

# **Local vs. Global Explainability**

<!-- External image removed -->

| Feature | Local Explainability | Global Explainability |
|---------|---------------------|----------------------|
| **Scope** | Individual predictions | Entire model behavior |
| **Question** | Why this specific prediction? | How does the model behave overall? |
| **Techniques** | LIME, SHAP, Counterfactuals | Permutation Importance, PDPs |
| **Strengths** | Precision, actionability | Big picture, auditing |
| **Limitations** | Limited scope, instability | Averages can mislead |

---

# **Best Practice: Combined Approach**

<!-- External image removed -->

- Start with **global explainability** to:
  - Understand overall patterns
  - Identify potential biases
  - Get big picture insights

- Then use **local explainability** to:
  - Analyze specific predictions
  - Understand edge cases
  - Provide reasoning for individual decisions

---

# **Practical Applications of XAI**

<!-- External image removed -->

- **Healthcare**: Explaining diagnostic recommendations
- **Finance**: Justifying loan approvals/denials
- **Autonomous Vehicles**: Understanding driving decisions
- **Legal Tech**: Supporting evidence analysis
- **Recommendation Systems**: Clarifying suggested items
- **HR**: Explaining candidate rankings

---

# **Implementation Considerations**

<div class="columns">
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">1</span> <strong>Define</strong> your explainability goals</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">2</span> <strong>Consider</strong> your audience</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">3</span> <strong>Select</strong> appropriate techniques</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">4</span> <strong>Balance</strong> accuracy vs. explainability</div>
</div>
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">5</span> <strong>Integrate</strong> explanation from the start</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">6</span> <strong>Test</strong> explanations with end users</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">7</span> <strong>Be aware</strong> of computational costs</div>
  
  <!-- External image removed -->
</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

# **Selecting the Right XAI Approach**

| If you need... | Consider... |
|---------------|-------------|
| Explanations for non-technical users | Counterfactual explanations, simple visualizations |
| Regulatory compliance | Global methods, comprehensive documentation |
| Debugging model behavior | Feature importance, PDPs, SHAP |
| Real-time explanations | Lightweight methods (simple trees, rule-based) |
| Detecting bias | Global methods across demographic groups |

---

# **Recent Advances in XAI**

- **Concept-based explanations**: Explaining in high-level human concepts
- **Self-explaining models**: Architectures designed to be inherently explainable
- **Causal XAI**: Moving beyond correlations to identify causal relationships
- **Explanations via natural language**: Generating human-like explanations
- **Interactive exploratory interfaces**: Allowing users to probe model behavior

---

# **XAI and Large Language Models**

<!-- External image removed -->

- **Chain-of-thought reasoning**
- **Attention visualization**
- **Prompt engineering for explainability**
- **Retrieval augmentation to trace sources**
- **Self-consistency checks**
- **Factuality evaluation**

---

# **Challenges in XAI Implementation**

- **Trade-off between accuracy and explainability**
- **Lack of standardized evaluation metrics**
- **Explanations may not align with human reasoning**
- **Computational overhead**
- **Security concerns with revealing model internals**
- **Rapidly evolving regulatory landscape**

---

# **XAI Evaluation Metrics**

<!-- External image removed -->

- **Fidelity**: How accurately explanations represent model behavior
- **Comprehensibility**: How easily humans understand explanations
- **Completeness**: How thoroughly explanations cover model decisions
- **Consistency**: How similar explanations are across similar inputs
- **Robustness**: How stable explanations are to small input changes

---

# **Ethical Considerations**

<!-- External image removed -->

- **Explanations can be misleading**
- **Psychological anchoring in explanations**
- **Balancing transparency vs. IP protection**
- **Different stakeholders need different explanations**
- **Potential for explanation manipulation**
- **"Right to explanation" legal frameworks**

---

# **Student Project Implementation Checklist**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2em; margin: 20px 0;">
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">1</span> <strong>Define</strong> your explainability requirements</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">2</span> <strong>Choose</strong> appropriate model architecture</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">3</span> <strong>Integrate</strong> XAI techniques early in development</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">4</span> <strong>Document</strong> your explainability approach</div>
</div>
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">5</span> <strong>Test</strong> explanations with target users</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">6</span> <strong>Iterate</strong> based on feedback</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">7</span> <strong>Consider</strong> computational requirements</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">8</span> <strong>Address</strong> ethical implications</div>
</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

# **Tools and Libraries for XAI**

<table class="feature-table">
<tr>
<td width="50%">
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>SHAP</strong>: <a href="https://github.com/slundberg/shap">https://github.com/slundberg/shap</a></div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>LIME</strong>: <a href="https://github.com/marcotcr/lime">https://github.com/marcotcr/lime</a></div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>InterpretML</strong>: <a href="https://interpret.ml">https://interpret.ml</a> (Microsoft)</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Captum</strong>: <a href="https://captum.ai">https://captum.ai</a> (Facebook)</div>
</td>
<td width="50%">
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>AIX360</strong>: <a href="https://aix360.mybluemix.net">https://aix360.mybluemix.net</a> (IBM)</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>ELI5</strong>: <a href="https://eli5.readthedocs.io">https://eli5.readthedocs.io</a></div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>What-If Tool</strong>: <a href="https://pair-code.github.io/what-if-tool">https://pair-code.github.io/what-if-tool</a> (Google)</div>
</td>
</tr>
</table>

<div style="text-align: center; margin-top: 20px;">
<!-- External image removed -->
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

# **Future Directions in XAI**

<div class="columns">
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Multi-modal explanations</strong> combining text, visuals, and interactive elements</div>
  
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Personalized explanations</strong> tailored to user expertise and needs</div>
  
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Standardization</strong> of XAI metrics and methods</div>
  
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Regulatory frameworks</strong> demanding higher levels of explainability</div>
  
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> <strong>Integration with causal inference</strong> for deeper understanding</div>
</div>

<div>
  <!-- External image removed -->
</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>

---

# **Conclusion**

<div class="columns">
<div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> XAI is <strong>essential</strong> for responsible AI deployment</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> Balances <strong>power and transparency</strong></div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> Requires thoughtful selection of <strong>techniques</strong> based on use case</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> Should be integrated from the <strong>beginning</strong> of ML projects</div>
  <div style="margin-bottom: 15px;"><span class="custom-bullet">•</span> Combines <strong>technical solutions</strong> with <strong>human-centered design</strong></div>
</div>
<div>
  <!-- External image removed -->
</div>
</div>

<div class="footer">Explainable AI Lecture Series - 2025</div>