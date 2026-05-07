# core/evaluator.py
from langsmith import evaluate
from langchain_core.prompts import ChatPromptTemplate
from core.llm_client import get_llm


eval_prompt = ChatPromptTemplate.from_template(
 """You are an expert evaluator of legal and compliance reports.

Query: {query}

Analysis Provided: {analysis}

Final Report:
{final_report}

Score the report from 1-10 on the following criteria:

1. **Accuracy** - How factually correct and well-grounded is the content?
2. **Completeness** - Does it cover all major obligations, clauses, and aspects?
3. **Clarity** - How clear, professional, and readable is the language?
4. **Usefulness** - Are the findings and recommendations actionable?
5. **Citation Quality** - Are sources and clauses properly referenced?
6. **Risk Assessment Quality** - How well does it identify, categorize (High/Medium/Low), and explain risks?

Return **only** a valid JSON object:

{{
 "accuracy": X,
 "completeness": X,
 "clarity": X,
 "usefulness": X,
 "citation_quality": X,
 "risk_assessment_quality": X,
 "overall_score": X,
 "feedback": "brief but constructive comment"
    }}"""
)
        

class DocForgeEvaluator:
    def __init__(self):
        self.llm = get_llm("gemini")   # or "gemini" for better judgment

    def evaluate_report(self, query: str, analysis: str, final_report: str) -> dict:
        chain = eval_prompt | self.llm
        result = chain.invoke({
            "query": query,
            "analysis": analysis[:2500],      # Limit to avoid token overflow
            "final_report": final_report
        })

        try:
            import json
            scores = json.loads(result.content.strip())
            return scores
        except:
            # Fallback
            return {
                "accuracy": 7,
                "completeness": 7,
                "clarity": 8,
                "usefulness": 7,
                "citation_quality": 6,
                "risk_assessment_quality": 7,
                "overall_score": 7.0,
                "feedback": "Failed to parse evaluation JSON."
            }


# Global evaluator instance
evaluator = DocForgeEvaluator()