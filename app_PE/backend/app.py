from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from solver import solve_planning

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/solve")
def solve(payload: dict):
    return solve_planning(payload)

@app.get("/health")
def health():
    return {"ok": True}