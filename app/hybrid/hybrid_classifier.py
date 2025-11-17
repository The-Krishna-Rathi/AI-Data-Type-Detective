class HybridClassifier:

    def __init__(self, profiler, rule_engine, ai_infer):
        self.profiler = profiler
        self.rule_engine = rule_engine
        self.ai_infer = ai_infer

    def classify_column(self, series, column_name):

        # 1. profiler insights
        profile = self.profiler.profile_column(series)

        # 2. rule-based detection
        rule_result = self.rule_engine.detect(series)

        # 3. LLM inference
        ai_result = self.ai_infer.infer_column_types(column_name, profile)

        # Combine with weighted voting
        final_type = self.weighted_vote(rule_result, ai_result)

        return {
            "column_name": column_name,
            "final_type": final_type,
            "rule_based": rule_result,
            "ai_inference": ai_result,
            "profile": profile
        }

    def weighted_vote(self, rule, ai):
        scores = {}

        # Rule engine usually reliable for numeric/date/boolean
        scores[rule["type"]] = rule["confidence"] * 0.6

        # AI is more reliable for complex reasoning
        scores[ai["inferred_type"]] = ai["confidence"] * 0.4

        # pick highest score
        return max(scores, key=scores.get)
