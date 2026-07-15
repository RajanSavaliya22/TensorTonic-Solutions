import builtins
def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    type_map = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

    result = []

    for i, record in enumerate(records):
        errors = []

        for col_schema in schema:
            col = col_schema["column"]

            # Missing column
            if col not in record:
                errors.append(f"{col}: missing")
                continue

            value = record[col]

            # Null check
            if value is None:
                if not col_schema.get("nullable", False):
                    errors.append(f"{col}: null")
                continue

            expected = col_schema["type"]

            # Type check
            if expected == "float":
                valid = type(value) in (int, float)
            else:
                valid = type(value) is type_map[expected]

            if not valid:
                errors.append(
                    f"{col}: expected {expected}, got {type(value).__name__}"
                )
                continue

            # Range check
            if expected in ("int", "float"):
                if "min" in col_schema and value < col_schema["min"]:
                    errors.append(f"{col}: out of range")
                    continue
                if "max" in col_schema and value > col_schema["max"]:
                    errors.append(f"{col}: out of range")

        result.append((i, len(errors) == 0, errors))

    return result