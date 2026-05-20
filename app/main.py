from fastapi import FastAPI
from app.routers import info, text, files, ui

app = FastAPI(
    title="NER Redaction API",
    description="PII detection and anonymization service for Russian texts",
    version="0.1.0",
    openapi_version="3.0.3",
)

app.include_router(info.router)
app.include_router(text.router)
app.include_router(files.router)
app.include_router(ui.router)
