def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    output = []
    for request in requests:
        u_id = request['user_id']
        if u_id in feature_store:
            output.append(
                feature_store[u_id] | request['online_features']
            )
        else:
            output.append(
                defaults | request['online_features']
            )
    return output