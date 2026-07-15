def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    groups = {}

    for idx, record in enumerate(records):
        key = tuple(record.get(col) for col in key_columns)

        if key not in groups:
            groups[key] = {
                "first_idx": idx,
                "record": record
            }
        else:
            if strategy == "last":
                groups[key]["record"] = record

            elif strategy == "most_complete":
                current = groups[key]["record"]

                current_none = sum(v is None for v in current.values())
                new_none = sum(v is None for v in record.values())

                # Replace only if strictly more complete.
                # Ties keep the first occurrence.
                if new_none < current_none:
                    groups[key]["record"] = record

            # "first": do nothing

    # Preserve order of first appearance of each unique key
    result = [
        info["record"]
        for _, info in sorted(groups.items(), key=lambda x: x[1]["first_idx"])
    ]

    return result