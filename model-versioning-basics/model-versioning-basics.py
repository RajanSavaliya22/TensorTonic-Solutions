def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    sorted_models = sorted(models, key=lambda x: x["timestamp"], reverse=True)
    sorted_models = sorted(sorted_models, key=lambda x: x["latency"])
    sorted_models = sorted(sorted_models, key=lambda x: x["accuracy"], reverse=True)

    return sorted_models[0]['name']