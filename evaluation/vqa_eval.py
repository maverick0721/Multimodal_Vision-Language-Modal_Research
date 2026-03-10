def accuracy(pred,gt):

    correct = sum(
        p == g for p,g in zip(pred,gt)
    )

    return correct / len(pred)