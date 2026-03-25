"""
MiroFish Backend - Flask应用工厂
"""

import os
import time
import uuid
import warnings
from threading import Lock

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 设置JSON编码：确保中文直接显示（而不是 \uXXXX 格式）
    # Flask >= 2.3 使用 app.json.ensure_ascii，旧版本使用 JSON_AS_ASCII 配置
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # 设置日志
    logger = setup_logger('mirofish')
    
    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend 启动中...")
        logger.info("=" * 50)
    
    # 启用CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 注册模拟进程清理函数（确保服务器关闭时终止所有模拟进程）
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已注册模拟进程清理函数")
    
    # 请求日志中间件
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"请求: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"请求体: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"响应: {response.status_code}")
        return response
    
    # 注册蓝图
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # 健康检查
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}

    @app.route('/api/health')
    def api_health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}

    # ---------------------------------------------------------------------
    # Compatibility API for TradeFish integration.
    #
    # TradeFish expects a minimal "swarm simulation" REST API:
    #   POST /api/simulate -> {"simulation_id": "..."}
    #   GET  /api/simulation/<id>/status -> {"status":"completed"}
    #   GET  /api/simulation/<id>/report -> { ...report... }
    #
    # The full MiroFish backend uses richer endpoints under:
    #   /api/simulation/*, /api/report/*, /api/graph/*
    #
    # For now we provide a lightweight, deterministic, in-memory implementation
    # so TradeFish can run end-to-end without 404s while we evolve the deeper
    # bridge to the full simulation pipeline.
    # ---------------------------------------------------------------------
    _tf_lock = Lock()
    _tf_sims: dict[str, dict] = {}

    def _tf_sentiment_from_query(query: str) -> tuple[int, int, int]:
        # Deterministic pseudo-sentiment based on query hash.
        h = sum(ord(c) for c in (query or "")) % 100
        bullish = 30 + (h % 41)          # 30..70
        bearish = 20 + ((h * 7) % 41)    # 20..60
        neutral = max(0, 100 - bullish - bearish)
        # If we over-allocated, renormalize into 100 total.
        total = bullish + bearish + neutral
        if total != 100:
            scale = 100 / max(total, 1)
            bullish = int(round(bullish * scale))
            bearish = int(round(bearish * scale))
            neutral = max(0, 100 - bullish - bearish)
        return bullish, bearish, neutral

    @app.post('/api/simulate')
    def tf_simulate():
        data = request.get_json(silent=True) or {}
        prediction_query = str(data.get("prediction_query") or "")
        sim_id = f"tf_{uuid.uuid4().hex[:12]}"

        bullish, bearish, neutral = _tf_sentiment_from_query(prediction_query)
        narratives = []
        if prediction_query:
            narratives.append("Swarm aggregated the provided context into a short-term bias.")
        cascades = []
        if abs(bullish - bearish) < 8:
            cascades.append("High disagreement: narrative flip risk elevated.")

        report = {
            "prediction": "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral"),
            "sentiment_distribution": {"bullish": bullish, "bearish": bearish, "neutral": neutral},
            "key_narratives": narratives,
            "cascade_triggers": cascades,
            "contrarian_signals": [],
            "token_cost": 0.0,
            "generated_at": time.time(),
        }

        with _tf_lock:
            _tf_sims[sim_id] = {"created_at": time.time(), "status": "completed", "report": report}

        return jsonify({"simulation_id": sim_id})

    @app.get('/api/simulation/<simulation_id>/status')
    def tf_sim_status(simulation_id: str):
        with _tf_lock:
            sim = _tf_sims.get(simulation_id)
        if not sim:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": sim.get("status", "unknown")})

    @app.get('/api/simulation/<simulation_id>/report')
    def tf_sim_report(simulation_id: str):
        with _tf_lock:
            sim = _tf_sims.get(simulation_id)
        if not sim:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"report": sim.get("report", {})})
    
    if should_log_startup:
        logger.info("MiroFish Backend 启动完成")
    
    return app

