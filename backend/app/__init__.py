"""
MiroFish Backend - Flask应用工厂
"""

import os
import time
import uuid
import warnings
from dataclasses import dataclass
from threading import Lock, Thread

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import Config
from .services.graph_builder import GraphBuilderService
from .services.ontology_generator import OntologyGenerator
from .services.simulation_manager import SimulationManager, SimulationStatus
from .services.simulation_runner import SimulationRunner, RunnerStatus
from .utils.llm_client import LLMClient
from .utils.logger import setup_logger, get_logger
from .models.project import ProjectManager, ProjectStatus
from .models.task import TaskManager, TaskStatus


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
    # TradeFish bridge (REAL MiroFish mode)
    #
    # TradeFish expects:
    #   POST /api/simulate
    #   GET  /api/simulation/<id>/status
    #   GET  /api/simulation/<id>/report
    #
    # We implement these by running the real MiroFish pipeline in a background
    # thread:
    #   seed text -> ontology -> Zep graph -> simulation prepare -> run -> LLM summary
    #
    # If ZEP/LLM configuration is missing or the pipeline fails, we fall back to
    # a lightweight deterministic report (so TradeFish never hard-breaks).
    # ---------------------------------------------------------------------

    @dataclass
    class _TFJob:
        simulation_id: str
        status: str  # created | preparing | running | completed | failed
        created_at: float
        updated_at: float
        report: dict | None = None
        error: str = ""
        mode: str = "real"  # real | fallback

    _tf_lock = Lock()
    _tf_jobs: dict[str, _TFJob] = {}

    def _tf_seed_text(data: dict) -> str:
        q = str(data.get("prediction_query") or "").strip()
        mats = data.get("seed_materials") or []
        parts: list[str] = []
        if q:
            parts.append("PREDICTION_QUERY:\n" + q)
        for m in mats:
            if isinstance(m, dict):
                t = str(m.get("type") or "context")
                c = str(m.get("content") or "")
                if c.strip():
                    parts.append(f"[{t}]\n{c}".strip())
        return "\n\n---\n\n".join(parts).strip()

    def _tf_fallback_report(prediction_query: str) -> dict:
        # Deterministic pseudo-sentiment based on query hash.
        h = sum(ord(c) for c in (prediction_query or "")) % 100
        bullish = 30 + (h % 41)  # 30..70
        bearish = 20 + ((h * 7) % 41)  # 20..60
        neutral = max(0, 100 - bullish - bearish)
        total = bullish + bearish + neutral
        if total != 100:
            scale = 100 / max(total, 1)
            bullish = int(round(bullish * scale))
            bearish = int(round(bearish * scale))
            neutral = max(0, 100 - bullish - bearish)
        return {
            "prediction": "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral"),
            "sentiment_distribution": {"bullish": bullish, "bearish": bearish, "neutral": neutral},
            "key_narratives": ["Fallback mode: deterministic estimate (real pipeline unavailable)."],
            "cascade_triggers": [],
            "contrarian_signals": [],
            "token_cost": 0.0,
            "generated_at": time.time(),
            "mode": "fallback",
        }

    def _tf_set(job: _TFJob) -> None:
        with _tf_lock:
            _tf_jobs[job.simulation_id] = job

    def _tf_get(simulation_id: str) -> _TFJob | None:
        with _tf_lock:
            return _tf_jobs.get(simulation_id)

    def _tf_worker(simulation_id: str, payload: dict) -> None:
        job = _tf_get(simulation_id)
        if not job:
            return
        try:
            prediction_query = str(payload.get("prediction_query") or "").strip()
            seed_text = _tf_seed_text(payload) or prediction_query

            # Basic config checks.
            if not Config.ZEP_API_KEY or not Config.LLM_API_KEY:
                job.mode = "fallback"
                job.status = "completed"
                job.updated_at = time.time()
                job.report = _tf_fallback_report(prediction_query)
                _tf_set(job)
                return

            # 1) Create project + store extracted text
            job.status = "preparing"
            job.updated_at = time.time()
            _tf_set(job)

            proj = ProjectManager.create_project(name=f"TradeFish {simulation_id}")
            proj.simulation_requirement = prediction_query or "TradeFish swarm simulation"
            proj.total_text_length = len(seed_text)
            ProjectManager.save_extracted_text(proj.project_id, seed_text)

            # 2) Ontology (LLM)
            onto = OntologyGenerator().generate(
                document_texts=[seed_text],
                simulation_requirement=proj.simulation_requirement,
                additional_context="TradeFish bridge: derive a social-simulation ontology for market sentiment forecasting.",
            )
            proj.ontology = onto
            proj.analysis_summary = onto.get("analysis_summary")
            proj.status = ProjectStatus.ONTOLOGY_GENERATED
            ProjectManager.save_project(proj)

            # 3) Build Zep graph (async task + poll)
            builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
            task_id = builder.build_graph_async(
                text=seed_text,
                ontology=onto,
                graph_name=f"TradeFish {simulation_id}",
                chunk_size=int(payload.get("chunk_size") or 500),
                chunk_overlap=int(payload.get("chunk_overlap") or 50),
                batch_size=int(payload.get("batch_size") or 3),
            )

            tm = TaskManager()
            graph_id: str | None = None
            for _ in range(600):  # ~10 minutes worst-case at 1s intervals
                t = tm.get_task(task_id)
                if t and t.status == TaskStatus.COMPLETED:
                    graph_id = (t.result or {}).get("graph_id")
                    break
                if t and t.status == TaskStatus.FAILED:
                    raise RuntimeError("graph_build_failed")
                time.sleep(1.0)

            if not graph_id:
                raise RuntimeError("graph_build_timeout")

            proj.graph_id = graph_id
            proj.status = ProjectStatus.GRAPH_COMPLETED
            ProjectManager.save_project(proj)

            # 4) Create + prepare simulation
            sm = SimulationManager()
            sim = sm.create_simulation(project_id=proj.project_id, graph_id=graph_id, enable_twitter=True, enable_reddit=True)
            sm.prepare_simulation(
                simulation_id=sim.simulation_id,
                simulation_requirement=proj.simulation_requirement or "",
                document_text=seed_text,
                use_llm_for_profiles=True,
                parallel_profile_count=int(payload.get("parallel_profile_count") or 3),
            )

            # 5) Run simulation (real multi-agent OASIS)
            job.status = "running"
            job.updated_at = time.time()
            _tf_set(job)

            max_rounds = int(payload.get("simulation_rounds") or payload.get("rounds") or 6)
            SimulationRunner.start_simulation(sim.simulation_id, platform="parallel", max_rounds=max_rounds)

            # Wait until runner completes.
            for _ in range(3600):  # up to 1h
                rs = SimulationRunner.get_run_state(sim.simulation_id)
                if rs and rs.runner_status in (RunnerStatus.COMPLETED, RunnerStatus.FAILED, RunnerStatus.STOPPED):
                    break
                time.sleep(2.0)

            rs = SimulationRunner.get_run_state(sim.simulation_id)
            if not rs or rs.runner_status != RunnerStatus.COMPLETED:
                raise RuntimeError("simulation_run_failed")

            # 6) Summarize into TradeFish report schema (LLM JSON)
            llm = LLMClient()
            sys_prompt = (
                "You are generating a compact trading-signal summary from a completed multi-agent social simulation. "
                "Return ONLY valid JSON with keys: prediction, sentiment_distribution {bullish,bearish,neutral} as integers summing to 100, "
                "key_narratives (list of strings), cascade_triggers (list), contrarian_signals (list), token_cost (number)."
            )
            user_payload = {
                "prediction_query": prediction_query,
                "simulation_id": sim.simulation_id,
                "graph_id": graph_id,
                "notes": "Use the simulation run state + action statistics to infer sentiment split and key narratives.",
                "run_state": rs.to_detail_dict() if hasattr(rs, "to_detail_dict") else {},
            }
            rep = llm.chat_json(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": str(user_payload)},
                ],
                temperature=0.2,
                max_tokens=1200,
            )

            rep["generated_at"] = time.time()
            rep["mode"] = "real"
            rep.setdefault("sentiment_distribution", {"bullish": 34, "bearish": 33, "neutral": 33})

            job.status = "completed"
            job.updated_at = time.time()
            job.report = rep
            _tf_set(job)

        except Exception as e:
            # Fall back to keep TradeFish moving, but mark mode clearly.
            prediction_query = str(payload.get("prediction_query") or "").strip()
            job.mode = "fallback"
            job.status = "completed"
            job.updated_at = time.time()
            job.error = f"{e.__class__.__name__}"
            rep = _tf_fallback_report(prediction_query)
            rep["error"] = job.error
            job.report = rep
            _tf_set(job)

    @app.post("/api/simulate")
    def tf_simulate():
        payload = request.get_json(silent=True) or {}
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        job = _TFJob(
            simulation_id=simulation_id,
            status="created",
            created_at=time.time(),
            updated_at=time.time(),
            report=None,
            mode="real",
        )
        _tf_set(job)
        Thread(target=_tf_worker, args=(simulation_id, payload), daemon=True).start()
        return jsonify({"simulation_id": simulation_id})

    @app.get("/api/simulation/<simulation_id>/status")
    def tf_sim_status(simulation_id: str):
        job = _tf_get(simulation_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": job.status, "mode": job.mode, "error": job.error})

    @app.get("/api/simulation/<simulation_id>/report")
    def tf_sim_report(simulation_id: str):
        job = _tf_get(simulation_id)
        if not job:
            return jsonify({"error": "not_found"}), 404
        if not job.report:
            return jsonify({"error": "not_ready", "status": job.status}), 409
        return jsonify({"report": job.report})
    
    if should_log_startup:
        logger.info("MiroFish Backend 启动完成")
    
    return app

