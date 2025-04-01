# CMPT 3830: Final Project Requirements

## Overview

The final project represents the culmination of your journey through the Machine Learning Operations (MLOps) lifecycle. Throughout the semester, you have progressed from initial machine learning experiments in Jupyter notebooks to a production-ready application with proper organization, versioning, APIs, containerization, and monitoring. The final project asks you to present this journey, demonstrate your working solution, and reflect on challenges, limitations, and future directions.

## Presentation Requirements

You will deliver a 15-minute presentation covering:

1. **Complete MLOps Journey**
   - Trace your progress from initial experimentation to production-ready application
   - Highlight key architectural decisions and design patterns implemented
   - Show the evolution of your codebase and infrastructure

2. **Implementation Challenges**
   - Discuss specific challenges encountered during each phase of implementation:
     - Project structuring and code organization
     - Data and model versioning with DVC and MLflow
     - API development with Flask
     - Containerization with Docker
     - Monitoring and observability implementation
   - Explain how you overcame these challenges and what you learned

3. **Transformation Narrative**
   - Explain the process of transforming your Jupyter notebook experiment into a production-ready project
   - Highlight improvements in reproducibility, maintainability, and scalability
   - Demonstrate how your solution follows MLOps best practices

4. **Current Limitations**
   - Critically analyze limitations in your current solution, including:
     - Data limitations (quality, quantity, biases, etc.)
     - Model limitations (performance, complexity, interpretability)
     - API limitations (throughput, latency, features)
     - Deployment limitations (scalability, reliability, costs)
   - Explain the impact of these limitations on your solution's overall effectiveness

## Demonstration Requirements

5. **Live Demo**
   - Present a live demonstration of your application running in Docker
   - Show the complete user journey through your application
   - Demonstrate key features and functionality
   - Show your monitoring dashboards with Prometheus and Grafana

6. **Deployment Options**
   - Demonstrate your containerized application running locally with Docker
   - Explain your deployment configuration and architecture

7. **Deployment Strategies**
   - Explain which deployment strategy would be most suitable for your ML application:
     - **Batch deployment**: Processing data in batches at scheduled intervals
     - **Real-time deployment**: Providing immediate predictions based on incoming requests
     - **Hybrid deployment**: Combining aspects of batch and real-time approaches
     - **Blue-green deployment**: Running parallel deployments for seamless updates
     - **Canary deployment**: Gradually rolling out updates to a subset of users
   - Justify your choice based on your specific use case, data characteristics, and user requirements

8. **Future Work and Scaling**
   - Present a roadmap for future development and improvement
   - Discuss how your application could be made accessible to multiple concurrent users
   - Outline potential enhancements to your model, infrastructure, and user experience
   - Describe how you would implement CI/CD pipelines for continued development

## Evaluation Criteria

Your final project grade consists of two components:

### Group Presentation (20% of Total Grade)

The group presentation component will be evaluated based on:

1. **Technical Implementation** (40% of presentation grade)
   - Code quality, organization, and documentation
   - Proper implementation of MLOps practices
   - Containerization effectiveness
   - API design and functionality

2. **Presentation and Communication** (30% of presentation grade)
   - Clarity of presentation
   - Demonstration effectiveness
   - Technical accuracy
   - Ability to explain complex concepts

3. **Critical Analysis** (20% of presentation grade)
   - Depth of reflection on challenges and limitations
   - Quality of future work proposals
   - Understanding of deployment strategies

4. **Innovation and Creativity** (10% of presentation grade)
   - Unique approaches to solving problems
   - Creative extensions beyond basic requirements
   - Novel applications or insights

### Individual Q&A (80% of Total Grade)

Each team member will participate in an individual Q&A session where they will be assessed on:
- Personal understanding of the MLOps concepts implemented
- Ability to explain technical decisions and their consequences
- Comprehension of the entire project lifecycle
- Knowledge of their specific contributions and the overall system architecture

## Submission

Your final submission should include:

1. GitHub repository link containing your complete MLOps project
2. Presentation slides (PDF format)
3. Brief report (2-3 pages) summarizing your implementation, challenges, and future work

Submit all materials by the deadline: **[Thursday, April 17th, 2025 1:00 PM]**

The presentations will be held during our Lab time on **[Thursday, April 17th, 2025 2:00 PM to 5:00 PM]**

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
