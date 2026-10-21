"""FastAPI service wrapping the graph query functions."""

from fastapi import FastAPI

app = FastAPI(title="feder-kg")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: add endpoints wrapping the query functions
