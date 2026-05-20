from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

_UI_PATH = Path(__file__).parent.parent / "static" / "ui.html"

@router.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))
