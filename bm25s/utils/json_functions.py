import json

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    

def dumps_with_builtin(d: dict, **kwargs) -> str:
    return json.dumps(d, **kwargs)

def dumps_with_orjson(d: dict, **kwargs) -> str:
    if kwargs.get("ensure_ascii", True):
        # Simulate `ensure_ascii=True` by escaping non-ASCII characters
        return orjson.dumps(d).decode("utf-8").encode("ascii", "backslashreplace").decode("utf-8")
    # Ignore other kwargs not supported by orjson
    return orjson.dumps(d).decode("utf-8")

if ORJSON_AVAILABLE:
    def dumps(d: dict, **kwargs) -> str:
        return dumps_with_orjson(d, **kwargs)
    loads = orjson.loads
else:
    def dumps(d: dict, **kwargs) -> str:
        return dumps_with_builtin(d, **kwargs)
    loads = json.loads


