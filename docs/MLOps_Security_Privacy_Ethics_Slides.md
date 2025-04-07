---
marp: true
theme: default
paginate: true
header: "MLOps: Security, Compliance, Ethics & Privacy"
footer: "CMPT 2500 - 2025"
style: |
  section {
    font-size: 1.5em;
  }
  h1 {
    color: #0066cc;
  }
  h2 {
    color: #2c87c5;
  }
  table {
    font-size: 0.7em;
  }
---

# **Security, Compliance, Ethics & Privacy in MLOps**

---

# **Agenda**

- The Imperative of Security, Compliance, and Ethics in MLOps
- Security Challenges in MLOps
- Specific Security Threats & Safeguarding Mechanisms
- Regulatory Compliance: GDPR and CCPA
- Ethical Dimensions of Data Management
- Addressing Bias in ML Systems
- Explainable AI (XAI)
- Privacy-Preserving Techniques
  - Differential Privacy
  - Homomorphic Encryption
  - Federated Learning
- Best Practices and Frameworks

---

# **Introduction: The Imperative**

- **MLOps**: Critical discipline streamlining the end-to-end lifecycle of ML models
- **Adoption Against a Backdrop of Threats**: Increasing sophistication of attacks
- **Three Foundational Pillars**:
  - Security: Protecting integrity, confidentiality, and availability
  - Compliance: Adhering to regulatory frameworks (GDPR, CCPA)
  - Ethics: Ensuring fairness, transparency, and accountability

---

# **Security Challenges in MLOps**

- **Vast Amounts of Sensitive Data**: Personal, financial, health information
- **Complex Automated Pipelines**: Multiple points of vulnerability
- **Attack Vectors**:
  - Data poisoning & model inversion attacks
  - Model theft & extraction
  - Adversarial attacks
  - Inference attacks
  - Prompt injection attacks (for LLMs)
- **Risk Magnitude**: Higher due to automation and scale
- **Traditional Security Mechanisms**: Often insufficient for AI-specific vulnerabilities

---

# **Regulatory Compliance: GDPR and CCPA**

- **GDPR (General Data Protection Regulation)**
  - Right to explanation of algorithmic decisions
  - Right to be forgotten
  - Data minimization and purpose limitation

- **CCPA (California Consumer Privacy Act)**
  - Right to know what personal data is collected
  - Right to delete personal information
  - Right to opt-out of the sale of personal data

---

# **Ethical Dimensions of Data Management**

- **Extending Beyond Regulatory Compliance**
- **Key Ethical Considerations**:
  - Fairness and bias mitigation
  - Transparency in data collection and use
  - Consent and informed participation
  - Accountability for algorithmic decisions
  - Long-term societal impact assessment
  - Privacy and data security preservation
  - Addressing potential for misuse or weaponization

---

# **Addressing Bias in ML Systems**

- **Types of Bias**:
  - Historical bias: When data reflects past societal biases
  - Representation bias: Underrepresentation of certain groups
  - Measurement bias: Varying accuracy across different populations
  - Aggregation bias: Inappropriately combining data from different groups
  - Evaluation bias: Benchmarking on non-representative populations
  - Automation bias: Favoring automated systems regardless of errors
- **Real-World Impact**: Healthcare disparities, facial recognition errors, discriminatory lending

---

# **Explainable AI (XAI)**

- **Addressing the "Black Box" Problem**
- **Key Components**:
  - Transparent model design
  - Post-hoc explanation methods
  - Interpretability tools
- **Business and Regulatory Impact**:
  - Building user trust
  - Meeting regulatory requirements (e.g., GDPR "right to explanation")
  - Enabling human oversight
  - Troubleshooting and improving model performance

---

# **Specific Security Threats in ML Systems**

- **Data Poisoning Attacks**: Injection of corrupted data to degrade model performance
- **Model Theft**: Stealing proprietary models via extraction techniques
- **Model Inversion**: Reconstructing sensitive training data from model outputs
- **Adversarial Attacks**: Subtle input manipulations causing incorrect predictions
- **Prompt Injection**: For LLMs, manipulating prompts to generate harmful content
- **Infrastructure Vulnerabilities**: Exploiting MLOps platforms (Azure ML, BigML, Google Vertex)
- **Resource Exhaustion**: Overwhelming ML systems with computational demands

---

# **Safeguarding ML Models: Methods & Technologies**

- **Strong Data Encryption**: Both at rest and in transit
- **Strict Access Controls**: Authentication mechanisms and IAM solutions
- **Secure Execution Environments (TEEs)**: Protected inference spaces
- **Network Segmentation**: Isolating MLOps workloads
- **Monitoring & Anomaly Detection**: Real-time security oversight
- **Model Watermarking**: Deterring intellectual property theft
- **Zero-Trust Security Model**: Verification for every access attempt
- **Privacy-Preserving Techniques**: Differential privacy, homomorphic encryption, federated learning

---

# **Privacy-Preserving Techniques**

Three major approaches to preserving privacy in ML systems:

1. **Differential Privacy**
2. **Homomorphic Encryption**
3. **Federated Learning**

---

# **Differential Privacy: Introduction**

- Mathematically rigorous framework to protect individual data confidentiality
- Allows extraction of useful statistics while preserving privacy
- **Core Principle**: Strategic addition of calibrated random noise

---

# **Differential Privacy: How It Works**

- **ε-differential privacy**: Mathematical guarantee limiting information leakage
  - Smaller ε = stronger privacy but lower accuracy
- **Mechanism**: Adds noise proportional to the query's sensitivity
- **Benefits over traditional anonymization**:
  - Quantifiable privacy guarantees
  - Protection against re-identification attacks

---

# **Differential Privacy: Key Properties**

- **Composability**: Privacy guarantees degrade predictably with multiple queries
- **Post-processing**: Cannot reduce privacy through post-processing
- **Group Privacy**: Protects correlated records within a dataset

---

# **Homomorphic Encryption: Introduction**

- Cryptographic technique allowing computation on encrypted data
- Results remain encrypted and can only be decrypted by the data owner
- Solves the problem: "How to analyze data you cannot see?"

---

# **Homomorphic Encryption: Types**

- **Partially Homomorphic (PHE)**: Supports one operation (addition OR multiplication)
  - Examples: RSA, ElGamal, Paillier cryptosystems
- **Somewhat Homomorphic (SHE)**: Limited number of operations
- **Fully Homomorphic (FHE)**: Arbitrary computations on encrypted data
  - Major breakthrough but computationally expensive
- **Leveled Fully Homomorphic (LFHE)**: Limited depth of operations

---

# **Federated Learning: Introduction**

- Decentralized approach to ML model training
- Data remains on local devices/servers
- Only model updates shared and aggregated
- Ideal for inherently distributed data scenarios

---

# **Federated Learning: Process**

1. **Initialization**: Server creates initial global model
2. **Distribution**: Model sent to participating devices/nodes
3. **Local Training**: Each node trains on local data
4. **Update Aggregation**: Local model updates combined (e.g., FedAvg algorithm)
5. **Iteration**: Process repeats until convergence

---

# **Federated Learning: Advantages**

- **Privacy Preservation**: Raw data never leaves devices
- **Reduced Data Transfer**: Only model updates transmitted
- **Real-time Learning**: Can incorporate fresh data continuously
- **Regulatory Compliance**: Helps meet data locality requirements
- **Enhanced with**: Differential privacy, secure aggregation, etc.

---

# **Comparison of Privacy-Preserving Techniques**

| Feature | Differential Privacy | Homomorphic Encryption | Federated Learning |
| :---- | :---- | :---- | :---- |
| **Data Location** | Central/distributed; noise added | Encrypted; processes in untrusted environments | Remains decentralized on local devices |
| **Computation** | Statistical analysis with noise | Computation on encrypted data | Collaborative ML training; only updates shared |
| **Privacy Guarantee** | Mathematically provable (ε) | Data remains encrypted | Raw data stays local; can combine with other techniques |

---

# **Comparison (Continued)**

| Feature | Differential Privacy | Homomorphic Encryption | Federated Learning |
| :---- | :---- | :---- | :---- |
| **Computational Overhead** | Generally lower | Can be high, especially for FHE | Primarily communication overhead |
| **Key Applications** | Government statistics, healthcare analytics | Secure cloud computing, healthcare, finance | Mobile devices, healthcare, IoT networks |
| **Limitations** | Accuracy/privacy tradeoff | Computational cost | Data heterogeneity, model poisoning risks |

---

# **Why Privacy Techniques Matter in MLOps**

- **Regulatory Compliance**: Meeting GDPR, CCPA, HIPAA requirements
- **Maintaining Consumer Trust**: Protecting sensitive data
- **Expanding Data Access**: Enabling use of previously inaccessible data
- **Competitive Advantage**: Processing sensitive data securely
- **Risk Mitigation**: Reducing breach impact and liability

---

# **Implementation Considerations**

- **Differential Privacy**:
  - Privacy budget management
  - Noise calibration based on sensitivity
  - Available libraries: Google's Differential Privacy, OpenDP

- **Homomorphic Encryption**:
  - Key management infrastructure
  - Performance optimization techniques
  - Libraries: Microsoft SEAL, IBM HElib

- **Federated Learning**:
  - Client orchestration
  - Secure aggregation mechanisms
  - Frameworks: TensorFlow Federated, PySyft

---

# **Tangible Risks of Neglecting Security, Compliance & Ethics**

- **Legal Consequences**:
  - Substantial financial penalties (up to 4% of annual global revenue for GDPR)
  - Lawsuits from affected individuals
  - Regulatory investigations
  - Potential criminal charges
  - Contract invalidation

- **Reputational Damage**:
  - Erosion of customer trust and loyalty
  - Negative publicity and public backlash
  - Loss of competitive advantage
  - Difficulty attracting top talent
  - Hesitancy from business partners

---

# **Societal Impact of Security & Ethics Failures**

- Perpetuation and amplification of societal inequalities
- Spread of misinformation and disinformation
- Erosion of public trust in AI technology
- Potential harm from flawed systems in critical sectors
- Economic consequences of job displacement
- Environmental impacts from unsustainable AI practices

---

# **Best Practices for MLOps Security, Compliance & Privacy**

- **Defense in Depth Strategy**: Multiple security layers
- **Privacy by Design**: Built-in, not bolted-on
- **Continuous Monitoring**: Detect anomalies and drift
- **Regular Auditing**: Security and compliance assessments
- **Transparent Documentation**: Model cards, data sheets
- **Privacy Impact Assessments**: Before new data uses
- **Responsible Disclosure**: Clear policies for vulnerabilities
- **Employee Training**: Security awareness and ethical guidelines
- **Zero-Trust Security Model**: Verify every access attempt
- **Model Watermarking**: Protect intellectual property

---

# **Data Governance: The Foundation**

- **Comprehensive Framework** for responsible data management
- **Key Components**:
  - Data cataloging and classification
  - Access control and usage policies
  - Data lineage tracking
  - Data quality assurance
  - Compliance monitoring
  - Privacy preservation mechanisms

---

# **Actionable Recommendations**

1. Develop a comprehensive MLOps security strategy
2. Establish clear policies for regulatory compliance
3. Integrate ethical considerations into every MLOps stage
4. Invest in Explainable AI (XAI) tools and techniques
5. Adopt established security frameworks and best practices
6. Prioritize data governance as a cornerstone of responsible MLOps
7. Foster a culture of security, compliance, and ethics in teams
8. Continuously monitor and evaluate MLOps practices

---

# **Conclusion**

- Security, compliance, ethics, and privacy are **foundational requirements** for MLOps
- Privacy-preserving techniques enable **responsible AI development**
- Organizations must balance:
  - Innovation and utility
  - Security and privacy
  - Compliance and ethics
- **Proactive approach** required to build sustainable, trusted AI systems
- Neglecting these aspects leads to **tangible legal, reputational, and societal consequences**

---

# **Thank You**

Questions?
