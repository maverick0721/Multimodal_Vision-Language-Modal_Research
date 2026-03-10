def speculative_decode(
    draft_model,
    target_model,
    tokens,
    steps=4
):

    draft = draft_model.generate(tokens,steps)

    verified = target_model.generate(draft,steps)

    return verified