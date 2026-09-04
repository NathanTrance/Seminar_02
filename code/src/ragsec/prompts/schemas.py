CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "malware_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["malicious", "benign"],
                    "description": "Whether the target package is malicious or benign",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0",
                },
                "behaviors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Behavior type observed in the code",
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Source lines supporting this behavior claim (e.g., L12-L15)",
                            },
                        },
                        "required": ["type", "evidence"],
                        "additionalProperties": False,
                    },
                    "description": "List of observed security-relevant behaviors",
                },
                "rationale": {
                    "type": "string",
                    "description": "Concise reasoning for the classification decision",
                },
            },
            "required": ["verdict", "confidence", "behaviors", "rationale"],
            "additionalProperties": False,
        },
    },
}
