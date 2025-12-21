import uvicorn
import os
from api import app

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print("\nДокументация API:")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print(f"OpenAPI спецификация: http://{host}:{port}/openapi.json")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )