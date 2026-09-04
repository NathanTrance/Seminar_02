SYSTEM_PROMPT = (
    "You are performing static source-code analysis.\n"
    "Do not assume behavior that is not supported by the provided source or retrieved evidence.\n"
    "Retrieved examples are references, not proof that the target is malicious.\n"
    "Classify the TARGET only.\n"
    "Every claimed target behavior must cite target source lines."
)


def build_zero_shot_messages(code: str, package_name: str | None = None) -> list[dict]:
    pkg_info = f"package: {package_name}\n" if package_name else ""
    user_content = (
        f"{pkg_info}"
        "Analyze the following Python code for malicious behavior.\n"
        "Respond with a JSON object containing: verdict (malicious/benign), "
        "confidence (0.0-1.0), behaviors (list of {type, evidence}), and rationale.\n\n"
        f"{code}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_rag_messages(
    code: str,
    retrieved_context: str,
    package_name: str | None = None,
    contrastive: bool = False,
) -> list[dict]:
    pkg_info = f"package: {package_name}\n" if package_name else ""
    if contrastive:
        rag_instruction = (
            "Below are retrieved MALICIOUS and BENIGN reference examples.\n"
            "Compare the target code against both groups.\n"
            "The presence of similar malicious examples may indicate risk, "
            "but the presence of similar benign examples should reduce false alarms.\n"
            "Classify the TARGET only.\n"
        )
    else:
        rag_instruction = (
            "Below are retrieved reference examples for context.\n"
            "Use them to inform your analysis, but classify the TARGET only.\n"
        )

    user_content = (
        f"{pkg_info}"
        f"{rag_instruction}\n"
        f"=== Retrieved Evidence ===\n"
        f"{retrieved_context}\n"
        f"=== Target Code ===\n"
        f"{code}\n"
        "Respond with a JSON object containing: verdict (malicious/benign), "
        "confidence (0.0-1.0), behaviors (list of {type, evidence}), and rationale."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
