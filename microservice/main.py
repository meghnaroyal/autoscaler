"""
main.py - Entry point
"""
import uvicorn
from api import app

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING CPU METRICS MICROSERVICE")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  http://localhost:8000/metrics/cpu")
    print("  GET  http://localhost:8000/health")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
