"""
핀토 스튜디오 AI 자동보정 웹 서비스
=====================================
운영 구조:
  [스튜디오]  python train.py train  → generator.pth 생성
              /admin 에서 generator.pth 업로드
  [고객]      / 에서 원본 업로드 → 보정본 ZIP 다운로드
"""

import os
import io
import zipfile
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
from flask import (Flask, request, jsonify, render_template,
                   send_file, redirect, url_for, abort)
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "pinto-secret-change-me")

ADMIN_KEY  = os.environ.get("ADMIN_KEY", "pinto2026")   # 환경변수로 변경 권장
WEIGHT_DIR = "weights"
GEN_PATH   = os.path.join(WEIGHT_DIR, "generator.pth")
META_PATH  = os.path.join(WEIGHT_DIR, "meta.json")
EXTS       = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

os.makedirs(WEIGHT_DIR, exist_ok=True)


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def allowed(name: str) -> bool:
    return Path(name).suffix.lower() in EXTS


def model_ready() -> bool:
    return os.path.exists(GEN_PATH)


def get_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def admin_required(f):
    """Authorization: Bearer <ADMIN_KEY> 헤더 또는 ?key= 쿼리 확인"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        key = (request.headers.get("Authorization", "").replace("Bearer ", "")
               or request.args.get("key", "")
               or request.form.get("admin_key", ""))
        if key != ADMIN_KEY:
            abort(403)
        return f(*args, **kwargs)
    return wrapper


def run_inference(file_bytes: bytes, tile_size: int = 256) -> bytes:
    """원본 이미지 bytes → 보정 결과 JPEG bytes"""
    import torch
    import train as tr

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 매 요청마다 로드하면 느림 → 앱 시작 시 캐시 (load_model_cache 참고)
    G = _get_model(device)
    img    = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
    result = tr.apply_generator(G, img, tile_size=tile_size, device=device)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ── 모델 캐시 (프로세스 수명 동안 유지) ─────────────────────────────────────
_model_cache = {"G": None, "loaded_at": None}


def _get_model(device: str):
    import torch
    import train as tr

    mtime = os.path.getmtime(GEN_PATH) if os.path.exists(GEN_PATH) else None
    if _model_cache["G"] is None or _model_cache["loaded_at"] != mtime:
        G = tr.UNetGenerator().to(device)
        G.load_state_dict(torch.load(GEN_PATH, map_location=device))
        G.eval()
        _model_cache["G"]         = G
        _model_cache["loaded_at"] = mtime
        print(f"[Model] generator.pth 로드 완료 (device={device})")
    return _model_cache["G"]


# ══════════════════════════════════════════════════════════════════
# 고객 라우트
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    meta = get_meta()
    return render_template("index.html",
                           model_ready=model_ready(),
                           meta=meta)


@app.route("/retouch", methods=["POST"])
def retouch():
    """
    고객 보정 요청
    단일 파일: files["image"]  → JSON {"image": base64}
    복수 파일: files["images"] → ZIP download
    """
    if not model_ready():
        return jsonify({"error": "현재 서비스 준비 중입니다. 잠시 후 다시 시도해주세요."}), 503

    meta      = get_meta()
    tile_size = meta.get("img_size", 256)

    # 복수 파일 → ZIP
    files = request.files.getlist("images")
    if files and files[0].filename:
        zip_buf = io.BytesIO()
        errors  = []
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                if not allowed(f.filename):
                    continue
                try:
                    result = run_inference(f.read(), tile_size)
                    zf.writestr(Path(f.filename).stem + "_retouched.jpg", result)
                except Exception as e:
                    errors.append(f.filename)
        zip_buf.seek(0)
        if errors:
            print(f"[Warn] 처리 실패: {errors}")
        return send_file(zip_buf, mimetype="application/zip",
                         as_attachment=True,
                         download_name="pinto_retouched.zip")

    # 단일 파일 → base64 JSON (미리보기용)
    single = request.files.get("image")
    if single and single.filename:
        try:
            result = run_inference(single.read(), tile_size)
            return jsonify({"image": base64.b64encode(result).decode()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "image 또는 images 필드가 없습니다."}), 400


# ══════════════════════════════════════════════════════════════════
# 관리자 라우트  (/admin?key=ADMIN_KEY)
# ══════════════════════════════════════════════════════════════════

@app.route("/admin")
@admin_required
def admin():
    meta = get_meta()
    return render_template("admin.html",
                           model_ready=model_ready(),
                           meta=meta,
                           admin_key=request.args.get("key", ""))


@app.route("/admin/upload-model", methods=["POST"])
@admin_required
def upload_model():
    """generator.pth 업로드 → 서버에 저장 (즉시 캐시 무효화)"""
    pth_file  = request.files.get("model")
    meta_file = request.files.get("meta")

    if not pth_file or not pth_file.filename.endswith(".pth"):
        return jsonify({"error": "generator.pth 파일을 선택하세요."}), 400

    # 업로드 전 유효성 검사
    try:
        import torch
        import train as tr
        data = torch.load(io.BytesIO(pth_file.read()), map_location="cpu")
        G = tr.UNetGenerator()
        G.load_state_dict(data)
    except Exception as e:
        return jsonify({"error": f"모델 파일이 올바르지 않습니다: {e}"}), 400

    # 저장
    torch.save(data, GEN_PATH)
    _model_cache["G"] = None   # 캐시 무효화

    meta_info = {}
    if meta_file and meta_file.filename.endswith(".json"):
        meta_info = json.load(meta_file)
        with open(META_PATH, "w") as f:
            json.dump(meta_info, f)
    elif not os.path.exists(META_PATH):
        with open(META_PATH, "w") as f:
            json.dump({"img_size": 256}, f)

    key = request.form.get("admin_key", "")
    return redirect(url_for("admin", key=key))


@app.route("/admin/status")
@admin_required
def admin_status():
    meta = get_meta()
    return jsonify({
        "model_ready": model_ready(),
        "meta": meta,
        "gen_path": GEN_PATH,
    })


if __name__ == "__main__":
    if model_ready():
        print(f"[Model] generator.pth 존재 — 서비스 준비됨")
    else:
        print(f"[Warn] generator.pth 없음. /admin 에서 모델을 업로드하세요")
    app.run(debug=True, host="0.0.0.0", port=5000)
