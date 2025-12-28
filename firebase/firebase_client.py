from __future__ import annotations
import os
import datetime as dt
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore

def init_firestore() -> Optional[firestore.Client]:

    path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not path:
        return None

    if not firebase_admin._apps:
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

def save_run(
    db: firestore.Client,
    uid: str,
    dataset_name: str,
    threshold: float,
    overall_risk: str,
    anomaly_count: int,
    max_score: float,
    series: Dict[str, Any],
) -> str:
    now = dt.datetime.utcnow()
    doc = {
        "uid": uid,
        "datasetName": dataset_name,
        "createdAt": now,
        "threshold": float(threshold),
        "overallRisk": overall_risk,
        "anomalyCount": int(anomaly_count),
        "maxScore": float(max_score),
        "series": series,
        "modelVersion": "hybrid-tae-mvp-v1",
    }
    ref = db.collection("runs").document()
    ref.set(doc)
    return ref.id

def maybe_create_alert(db: firestore.Client, uid: str, run_id: str, overall_risk: str) -> Optional[str]:
    if overall_risk != "High":
        return None
    now = dt.datetime.utcnow()
    message = "High risk anomaly detected. Recommend field inspection + corrective action."
    alert = {
        "uid": uid,
        "runId": run_id,
        "risk": overall_risk,
        "message": message,
        "createdAt": now,
        "status": "new",
    }
    ref = db.collection("alerts").document()
    ref.set(alert)
    return ref.id

def list_runs(db: firestore.Client, uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    qs = (
        db.collection("runs")
        .where("uid", "==", uid)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    out = []
    for d in qs:
        doc = d.to_dict()
        doc["_id"] = d.id
        out.append(doc)
    return out

def list_alerts(db: firestore.Client, uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    qs = (
        db.collection("alerts")
        .where("uid", "==", uid)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    out = []
    for d in qs:
        doc = d.to_dict()
        doc["_id"] = d.id
        out.append(doc)
    return out
