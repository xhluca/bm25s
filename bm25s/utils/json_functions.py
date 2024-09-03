import json

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    

def dumps_with_builtin(d: dict) -> str:
    return json.dumps(d)

def dumps_with_orjson(d: dict) -> str:
    return orjson.dumps(d).decode('utf-8')

if ORJSON_AVAILABLE:
    dumps = dumps_with_orjson
    loads = orjson.loads
else:
    dumps = dumps_with_builtin
    loads = json.loads

