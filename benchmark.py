import requests
import time
import statistics

BASE_URL = "http://localhost:8000"  # change if different port

TEST_EMAIL = "benchmark_user@test.com"
TEST_PASSWORD = "testpassword123"
TEST_NAME = "Benchmark User"

def measure(label, fn, runs=10):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        resp = fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    avg = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    print(f"{label:<40} avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  status={resp.status_code}")
    return resp

def main():
    # register (or ignore if already exists)
    requests.post(f"{BASE_URL}/register", json={
        "email": TEST_EMAIL, "password": TEST_PASSWORD, "name": TEST_NAME
    })

    # login and grab token + user_id
    r = requests.post(f"{BASE_URL}/login", json={
        "email": TEST_EMAIL, "password": TEST_PASSWORD
    })
    if r.status_code != 200:
        print(f"Login failed: {r.text}")
        return

    token = r.json()["access_token"]
    user_id = r.json()["user_id"]
    headers = {"Authorization": f"Bearer {token}"}

    print(f"\nRunning benchmarks against {BASE_URL} (10 runs each)\n")
    print("-" * 70)

    measure("GET /",
            lambda: requests.get(f"{BASE_URL}/"))

    measure("POST /login",
            lambda: requests.post(f"{BASE_URL}/login", json={
                "email": TEST_EMAIL, "password": TEST_PASSWORD
            }))

    measure("GET /users/{id}",
            lambda: requests.get(f"{BASE_URL}/users/{user_id}", headers=headers))

    measure("GET /users/{id}/sessions",
            lambda: requests.get(f"{BASE_URL}/users/{user_id}/sessions", headers=headers))

    print("-" * 70)
    print("\nNote: /analyze-video is excluded (CV processing takes 10-30s by design)")

if __name__ == "__main__":
    main()
