import nltk


def tensor_to_tokens(t):

    # convert tensor → python list
    tokens = t.detach().cpu().tolist()

    # remove padding tokens (0)
    tokens = [str(x) for x in tokens if x != 0]

    return tokens


def bleu_score(pred, ref):

    scores = []

    for p, r in zip(pred, ref):

        p_tokens = tensor_to_tokens(p)
        r_tokens = tensor_to_tokens(r)

        reference = [r_tokens]

        score = nltk.translate.bleu_score.sentence_bleu(
            reference,
            p_tokens
        )

        scores.append(score)

    if len(scores) == 0:
        return 0

    return sum(scores) / len(scores)