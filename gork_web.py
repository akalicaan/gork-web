
# -*- coding: utf-8 -*-
"""
Flask Web sürümü – Roulette Çarkı (Ali / gork)
- Masaüstü Tkinter yerine web arayüzü.
- Orijinal DealerModel, geçiş matrisi, bounce/ewma, regresyon düzeltmeleri korunmuştur.
- Basit, mobil uyumlu arayüz (index.html).
"""

import json, math, time, os
from collections import deque
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import numpy as np
from scipy.stats import entropy
from flask import Flask, jsonify, request, render_template, send_file

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(APP_DIR, "roulette_models.json")

# ---------- Wheel mapping (European single-zero) ----------
WHEEL_NUM_TO_POS = {
    0: 0, 26: 1, 3: 2, 35: 3, 12: 4, 28: 5, 7: 6, 29: 7, 18: 8, 22: 9, 9: 10,
    31: 11, 14: 12, 20: 13, 1: 14, 33: 15, 16: 16, 24: 17, 5: 18, 10: 19,
    23: 20, 8: 21, 30: 22, 11: 23, 36: 24, 13: 25, 27: 26, 6: 27, 34: 28,
    17: 29, 25: 30, 2: 31, 21: 32, 4: 33, 19: 34, 15: 35, 32: 36
}
WHEEL_POS_TO_NUM = {v: k for k, v in WHEEL_NUM_TO_POS.items()}
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

def num_color(n: int) -> str:
    if n == 0: return "green"
    return "red" if n in RED_NUMBERS else "black"

def neighbors_8_1_8(num: int):
    pos = WHEEL_NUM_TO_POS[num]
    left = [(pos - i) % 37 for i in range(1, 9)]
    right = [(pos + i) % 37 for i in range(1, 9)]
    return [WHEEL_POS_TO_NUM[p] for p in left], [num], [WHEEL_POS_TO_NUM[p] for p in right]

def modulo37_dist(p_from: int, p_to: int) -> int:
    dist = (p_to - p_from) % 37
    if dist > 18:
        dist -= 37
    return dist

# ---------- Gelişmiş Dealer Model ----------
class DealerModel:
    def _update_regression(self, laps: float, bounce_diff: float):
        # Online OLS with ridge regularization (lambda=0.1)
        lambda_ridge = 0.1
        self._reg_n += 1
        self._Sx += laps
        self._Sy += bounce_diff
        self._Sxx += laps * laps
        self._Sxy += laps * bounce_diff
        denom = (self._reg_n * self._Sxx - self._Sx * self._Sx) + lambda_ridge * self._reg_n
        if abs(denom) < 1e-9:
            return 0.0
        a = (self._reg_n * self._Sxy - self._Sx * self._Sy) / denom
        b = (self._Sy - a * self._Sx) / (self._reg_n + lambda_ridge)
        return a * laps + b

    def _regression_predict(self, laps: float) -> float:
        if self._reg_n < 5:
            return 0.0
        denom = (self._reg_n * self._Sxx - self._Sx * self._Sx) + 0.1 * self._reg_n
        if abs(denom) < 1e-9:
            return 0.0
        a = (self._reg_n * self._Sxy - self._Sx * self._Sy) / denom
        b = (self._Sy - a * self._Sx) / (self._reg_n + 0.1)
        return a * laps + b

    def __init__(self, name: str, prior_alpha: float = 1.0, smoothing: float = 0.1, bounce_alpha: float = 0.05, decay: float = 0.995):
        self.default_laps = 15.5
        self._reg_n = 0
        self._Sx = 0.0
        self._Sy = 0.0
        self._Sxx = 0.0
        self._Sxy = 0.0

        self.name = name
        self.prior_alpha = prior_alpha
        self.smoothing = smoothing
        self.bounce_alpha = bounce_alpha
        self.decay = decay
        # 1. derece: Offset counts
        self.prior_weights = [prior_alpha] * 37
        self.observed_counts = np.zeros(37)
        self.weights = np.array(self.prior_weights) + self.smoothing
        # 3. derece: Transition matrix (last2_k, last_k -> k)
        self.transition_prior = 0.1
        self.transition_counts = np.full((37, 37, 37), self.transition_prior)
        # State
        self.last_number = None
        self.last_k = None
        self.last2_k = None  # Ek: 3. derece için
        self.history = deque(maxlen=2048)
        self.recent_bounces = deque(maxlen=10)  # Artırıldı
        self.observation_count = 0
        self.last_was_lose = False
        self.lap_times = deque(maxlen=20)
        self.reverse_streak = 0  # Ek: Reverse için

    def to_json(self):
        return {
            "name": self.name, "prior_alpha": self.prior_alpha, "smoothing": self.smoothing,
            "bounce_alpha": self.bounce_alpha, "decay": self.decay,
            "prior_weights": self.prior_weights, "observed_counts": self.observed_counts.tolist(),
            "transition_counts": self.transition_counts.tolist(),
            "last_number": self.last_number, "last_k": self.last_k, "last2_k": self.last2_k,
            "history": list(self.history), "recent_bounces": list(self.recent_bounces),
            "observation_count": self.observation_count, "last_was_lose": self.last_was_lose,
            "default_laps": self.default_laps,
            "_reg_n": self._reg_n, "_Sx": self._Sx, "_Sy": self._Sy, "_Sxx": self._Sxx, "_Sxy": self._Sxy,
            "lap_times": list(self.lap_times)
        }

    @classmethod
    def from_json(cls, data):
        m = cls(data["name"], data.get("prior_alpha",1.0), data.get("smoothing",0.1), data.get("bounce_alpha", 0.05), data.get("decay",0.995))
        m.prior_weights = list(data.get("prior_weights", [1.0]*37))
        m.observed_counts = np.array(data.get("observed_counts", [0.0]*37))
        m.weights = np.array(m.prior_weights) + m.observed_counts + m.smoothing
        m.transition_counts = np.array(data.get("transition_counts", np.full((37,37,37), 0.1)))
        m.last_number = data.get("last_number")
        m.last_k = data.get("last_k")
        m.last2_k = data.get("last2_k")
        m.history = deque(data.get("history", []), maxlen=2048)
        m.recent_bounces = deque(data.get("recent_bounces", []), maxlen=10)
        m.observation_count = data.get("observation_count", 0)
        m.last_was_lose = data.get("last_was_lose", False)
        m.default_laps = data.get("default_laps", 15.5)
        m._reg_n = data.get("_reg_n", 0)
        m._Sx = data.get("_Sx", 0.0)
        m._Sy = data.get("_Sy", 0.0)
        m._Sxx = data.get("_Sxx", 0.0)
        m._Sxy = data.get("_Sxy", 0.0)
        m.lap_times = deque(data.get("lap_times", []), maxlen=20)
        return m

    def observe_spin(self, new_number: int, was_lose: bool = False, lap_time: float = None, laps: float = None):
        if lap_time is not None:
            self.lap_times.append(lap_time)
        if laps is not None:
            self.default_laps = laps  # Kullanıcıdan güncelle
        if self.last_number is None:
            self.last_number = new_number
            return
        p_from = WHEEL_NUM_TO_POS[self.last_number]
        p_to = WHEEL_NUM_TO_POS[new_number]
        k = (p_to - p_from) % 37

        # Decay uygula
        self.observed_counts *= self.decay
        self.transition_counts *= self.decay

        # Update
        self.observed_counts[k] += 1.0
        if self.last_k is not None and self.last2_k is not None:
            self.transition_counts[self.last2_k, self.last_k, k] += 1.0
        self.last2_k = self.last_k
        self.last_k = k

        # LOSE ise agresif update
        if was_lose:
            self.observed_counts[k] *= 1.5
            if self.last_k is not None and self.last2_k is not None:
                self.transition_counts[self.last2_k, self.last_k, k] *= 1.5
            self.last_was_lose = True
        else:
            self.last_was_lose = False

        self.weights = np.array(self.prior_weights) + self.observed_counts + self.smoothing

        # Bounce hesapla
        ham_offset = self.predict_offset()
        ham_pos = (p_from + ham_offset) % 37
        bounce_diff = modulo37_dist(ham_pos, p_to)
        laps_used = laps or self.default_laps
        self._update_regression(laps_used, float(bounce_diff))

        # Outlier filtre: abs >18 ise ekleme
        if abs(bounce_diff) <= 18:
            self.recent_bounces.append(bounce_diff)
        self.observation_count += 1

        self.history.append((self.last_number, new_number, k))
        self.last_number = new_number

    def predict_offset(self):
        if self.observation_count < 3 or self.last_k is None or self.last2_k is None:
            # Erken: 1. derece
            total = np.sum(self.weights)
            probs = self.weights / total if total > 0 else np.full(37, 1/37)
        else:
            # 3. derece
            slice_2d = self.transition_counts[self.last2_k, self.last_k]
            total = np.sum(slice_2d)
            probs = slice_2d / total if total > 0 else np.full(37, 1/37)
        return int(np.argmax(probs))

    def predict(self):
        if self.last_number is None:
            return None, None, 0, 0

        pos = WHEEL_NUM_TO_POS[self.last_number]
        ham_offset = self.predict_offset()
        ham_pos = (pos + ham_offset) % 37
        ham_num = WHEEL_POS_TO_NUM[ham_pos]

        # Bounce: EWMA son 10
        if len(self.recent_bounces) >= 1:
            weights = [self.decay ** i for i in range(len(self.recent_bounces))]
            weights = weights[::-1]  # En yeni en ağır
            bounce = sum(w * d for w, d in zip(weights, self.recent_bounces)) / sum(weights)
        else:
            bounce = 0

        # Regression düzeltme
        laps_used = self.default_laps
        reg_corr = self._regression_predict(laps_used)
        bounce += reg_corr

        # Bias corr: Ortalama miss +0.9
        bounce += 0.9 * min(1.0, self.observation_count / 50)

        # Hız etkisi
        if len(self.lap_times) > 0:
            avg_lap = np.mean(self.lap_times)
            if avg_lap > 0.75:
                bounce += 2
            elif avg_lap < 0.60:
                bounce -= 2

        adj_pos = (ham_pos + round(bounce)) % 37
        adj_num = WHEEL_POS_TO_NUM[adj_pos]

        return ham_num, adj_num, ham_offset, round(bounce)

    def confidence(self) -> float:
        total = np.sum(self.weights)
        if total <= 0: return 0.0
        probs = self.weights / total
        a, b = sorted(probs, reverse=True)[:2]
        gap = a - b
        ent = entropy(probs + 1e-10) / np.log(37)
        conf_gap = 0.5 * gap
        conf_ent = 1 - ent
        # Ağırlık değiştir: ent daha ağır
        data_factor = min(1.0, self.observation_count / 20)
        conf = max(0.3 if self.observation_count < 12 else 0.0, 
                   conf_gap * 0.4 + conf_ent * 0.6) * data_factor
        return float(conf)

# ---------- Persistence ----------
def save_models(models: dict, active: str, path: str = STORE_PATH):
    data = {"active": active, "models": {n: m.to_json() for n,m in models.items()}, "timestamp": time.time()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_models(path: str = STORE_PATH):
    if not os.path.exists(path): return {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = {n: DealerModel.from_json(md) for n, md in data.get("models", {}).items()}
        return models, data.get("active")
    except Exception:
        return {}, None

def parse_text_to_numbers(text: str):
    nums = []
    cur = ""
    for ch in text:
        if ch.isdigit():
            cur += ch
        else:
            if cur != "":
                try:
                    n = int(cur)
                    if 0 <= n <= 36:
                        nums.append(n)
                except:
                    pass
                cur = ""
    if cur != "":
        try:
            n = int(cur)
            if 0 <= n <= 36:
                nums.append(n)
        except:
            pass
    return nums

# ---------- App State (Web) ----------
app = Flask(__name__)

# Global durum (tek kullanıcı için basit tutuyoruz; çok kullanıcıda session/DB gerekir)
models, active_name = load_models()
if not models:
    models["Krupiye-1"] = DealerModel("Krupiye-1")
active_dealer_name = active_name if active_name in models else list(models.keys())[0]
recent_spins = deque(maxlen=5)
pending_prediction: Optional[Dict[str,Any]] = None
session_log: List[Dict[str,Any]] = []
initial_bet = 0.0
current_bet = 0.0
total_profit = 0.0
bankroll = 0.0
reverse_mode = False
bulk_mode = False
undo_stack: List[Dict[str,Any]] = []
bet_active = False

def get_active() -> DealerModel:
    return models[active_dealer_name]

def get_consecutive_losses():
    consecutive = 0
    for rec in session_log:
        if rec["result"] == "LOSE":
            consecutive += 1
        else:
            break
    return consecutive

def compute_bet_gate():
    K = 20
    wins = sum(1 for rec in session_log[:K] if rec["result"] == "WIN")
    total = sum(1 for rec in session_log[:K] if rec["result"] in ("WIN","LOSE"))
    hit_rate = (wins/total) if total>0 else 0.0
    model = get_active()
    conf = model.confidence()
    if pending_prediction:
        conf = pending_prediction['conf']
    enough_spins = len(model.history) >= 30  # Artırıldı
    consecutive_losses = get_consecutive_losses()
    streak_safe = consecutive_losses < 3
    thr = 0.40 - 0.20*conf  # Düşürüldü
    ok = (hit_rate >= thr) and enough_spins and streak_safe
    return ok, hit_rate, thr, conf, len(model.history), total, consecutive_losses

def make_new_prediction():
    global pending_prediction
    model = get_active()
    if model.last_number is None:
        pending_prediction = None
        return
    ham_num, adj_num_temp, ham_offset, bounce = model.predict()
    if ham_num is None:
        pending_prediction = None
        return

    pos = WHEEL_NUM_TO_POS[model.last_number]
    ham_pos = (pos + ham_offset) % 37

    # Bounce uygula, reverse_mode ise ters
    bounce_adjusted = round(bounce) if not reverse_mode else -round(bounce)
    adj_pos = (ham_pos + bounce_adjusted) % 37
    adj_num = WHEEL_POS_TO_NUM[adj_pos]

    left, center, right = neighbors_8_1_8(adj_num)
    covered = set(left + center + right)
    conf = model.confidence()
    pending_prediction = {
        "when": time.time(),
        "dealer": active_dealer_name,
        "last": model.last_number,
        "adj_pred": adj_num,
        "conf": float(conf),
        "covered": sorted(list(covered)),
        "left": left,
        "center": center,
        "right": right,
        "result": None,
        "actual": None
    }

def resolve_pending_with(actual_num: int):
    global pending_prediction, reverse_mode, total_profit, bankroll, current_bet, bet_active
    if not pending_prediction:
        return
    p = pending_prediction
    if p["result"] is not None:
        return

    win = actual_num in set(p["covered"])
    p["result"] = "WIN" if win else "LOSE"
    p["actual"] = actual_num

    if not bulk_mode:
        session_log.insert(0, p.copy())

    # Martingale kısmını basit tutuyoruz (orijinal mantıkla)
    if bet_active and initial_bet > 0:
        base_bet = float(current_bet)
        if win:
            total_profit += 1.90
            bankroll += 1.90
            current_bet = initial_bet
        else:
            total_profit -= 1.70
            bankroll -= 1.70
            consecutive_losses = get_consecutive_losses()
            current_bet = round(initial_bet * (2 ** min(consecutive_losses, 5)), 2)

    # Akıllı Reverse: iki miss negatif/pozitif ise tersle
    if p["result"] == "LOSE":
        model = get_active()
        ham_offset = model.predict_offset()
        p_from = WHEEL_NUM_TO_POS[p["last"]]
        ham_pos = (p_from + ham_offset) % 37
        actual_pos = WHEEL_NUM_TO_POS[actual_num]
        miss_dist = modulo37_dist(ham_pos, actual_pos)
        if len(session_log) >= 1 and session_log[0]["result"] == "LOSE":
            prev = session_log[0]
            prev_miss = modulo37_dist(WHEEL_NUM_TO_POS[prev["last"]] + model.predict_offset(), WHEEL_NUM_TO_POS[prev["actual"]])
            if miss_dist < 0 and prev_miss < 0:
                reverse_mode = True
            elif miss_dist > 0 and prev_miss > 0:
                reverse_mode = False
        if reverse_mode and get_consecutive_losses() >= 2:
            reverse_mode = False

    pending_prediction = None

def snapshot_state():
    try:
        models_dump = {n: m.to_json() for n, m in models.items()}
    except Exception:
        models_dump = {}
    snap = {
        "models": models_dump,
        "active": active_dealer_name,
        "recent_spins": list(recent_spins),
        "pending_prediction": json.loads(json.dumps(pending_prediction)) if pending_prediction else None,
        "session_log": json.loads(json.dumps(session_log)),
        "initial_bet": initial_bet,
        "current_bet": current_bet,
        "total_profit": total_profit,
        "bankroll": bankroll,
        "bet_active": bet_active,
        "reverse_mode": reverse_mode,
    }
    undo_stack.append(snap)
    if len(undo_stack) > 100:
        undo_stack.pop(0)

def restore_state(snap):
    global models, active_dealer_name, recent_spins, pending_prediction, session_log
    global initial_bet, current_bet, total_profit, bankroll, bet_active, reverse_mode
    try:
        models = {n: DealerModel.from_json(md) for n, md in snap.get("models", {}).items()}
    except Exception:
        pass
    active_dealer_name = snap.get("active", active_dealer_name)
    from collections import deque as _dq
    recent_spins = _dq(snap.get("recent_spins", []), maxlen=5)
    pending_prediction = snap.get("pending_prediction", None)
    session_log = snap.get("session_log", [])
    initial_bet = snap.get("initial_bet", initial_bet)
    current_bet = snap.get("current_bet", current_bet)
    total_profit = snap.get("total_profit", total_profit)
    bankroll = snap.get("bankroll", bankroll)
    bet_active = snap.get("bet_active", bet_active)
    reverse_mode = snap.get("reverse_mode", False)

def state_payload():
    ok, hit_rate, thr, conf, nsp, total, streak = compute_bet_gate()
    model = get_active()
    if not pending_prediction and model.last_number is not None:
        make_new_prediction()
    return {
        "dealer": active_dealer_name,
        "last_number": model.last_number,
        "pending": pending_prediction,
        "recent_spins": list(recent_spins),
        "session_log": session_log[:50],
        "stats": {
            "wins": sum(1 for rec in session_log if rec["result"] == "WIN"),
            "losses": sum(1 for rec in session_log if rec["result"] == "LOSE"),
            "streak": get_consecutive_losses(),
            "hit_rate": hit_rate,
            "thr": thr,
            "conf": conf,
            "spins": len(model.history)
        },
        "martingale": {
            "initial_bet": initial_bet,
            "current_bet": current_bet,
            "profit": total_profit,
            "bankroll": bankroll
        },
        "flags": {
            "bet_active": bet_active if bet_active else ok,
            "reverse_mode": reverse_mode,
            "gate_ok": ok
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state")
def api_state():
    return jsonify(state_payload())

@app.route("/api/bet_activate", methods=["POST"])
def api_bet_activate():
    global bet_active
    streak = get_consecutive_losses()
    if streak >= 3:
        bet_active = False
    else:
        bet_active = True
    return jsonify(state_payload())

@app.route("/api/undo", methods=["POST"])
def api_undo():
    if not undo_stack:
        return jsonify({"ok": False, "msg": "Geri alınacak bir işlem yok."}), 400
    snap = undo_stack.pop()
    restore_state(snap)
    return jsonify(state_payload())

@app.route("/api/add_spin", methods=["POST"])
def api_add_spin():
    data = request.get_json(force=True)
    n = int(data.get("n"))
    lap_time = float(data["lap_time"]) if data.get("lap_time") not in (None, "",) else None
    laps = float(data["laps"]) if data.get("laps") not in (None, "",) else None
    snapshot_state()
    was_lose = pending_prediction and pending_prediction.get("result") == "LOSE" if pending_prediction else False
    if pending_prediction:
        resolve_pending_with(n)
    model = get_active()
    model.observe_spin(n, was_lose=was_lose, lap_time=lap_time, laps=laps)
    recent_spins.appendleft(n)
    if len(recent_spins) > recent_spins.maxlen:
        recent_spins.pop()
    save_models(models, active_dealer_name, STORE_PATH)
    make_new_prediction()
    return jsonify(state_payload())

@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    data = request.get_json(force=True)
    n = int(data.get("n"))
    snapshot_state()
    resolve_pending_with(n)
    save_models(models, active_dealer_name, STORE_PATH)
    return jsonify(state_payload())

@app.route("/api/set_initial_bet", methods=["POST"])
def api_set_initial_bet():
    global initial_bet, current_bet
    data = request.get_json(force=True)
    val = round(float(str(data.get("amount")).replace(",", ".")), 2)
    if val <= 0:
        return jsonify({"ok": False, "msg": "Geçerli bir miktar gir."}), 400
    initial_bet = val
    current_bet = val
    return jsonify(state_payload())

@app.route("/api/set_bankroll", methods=["POST"])
def api_set_bankroll():
    global bankroll
    data = request.get_json(force=True)
    bankroll = float(str(data.get("amount")).replace(",", "."))
    return jsonify(state_payload())

@app.route("/api/set_laps", methods=["POST"])
def api_set_laps():
    data = request.get_json(force=True)
    v = float(str(data.get("value")).replace(",", "."))
    get_active().default_laps = max(5.0, min(40.0, v))
    return jsonify(state_payload())

@app.route("/api/toggle_reverse", methods=["POST"])
def api_toggle_reverse():
    global reverse_mode
    reverse_mode = not reverse_mode
    return jsonify(state_payload())

@app.route("/api/bulk_import", methods=["POST"])
def api_bulk_import():
    global bulk_mode
    data = request.get_json(force=True)
    raw = data.get("text","")
    top_is_latest = bool(data.get("top_is_latest", False))
    nums = parse_text_to_numbers(raw)
    if not nums:
        return jsonify({"ok": False, "msg": "Geçerli sayı yok."}), 400
    if top_is_latest:
        nums = list(reversed(nums))
    nums = list(reversed(nums))
    bulk_mode = True
    for n in nums:
        # aynı mantık: resolve, observe
        if pending_prediction:
            resolve_pending_with(n)
        model = get_active()
        model.observe_spin(n)
        recent_spins.appendleft(n)
        if len(recent_spins) > recent_spins.maxlen:
            recent_spins.pop()
    bulk_mode = False
    save_models(models, active_dealer_name, STORE_PATH)
    make_new_prediction()
    return jsonify(state_payload())

@app.route("/download/log.txt")
def download_log():
    lines = []
    if pending_prediction:
        p = pending_prediction
        lines.append(f"[BEKLEME] {p['dealer']} | Son:{p['last']} | Tahmin:{p['adj_pred']} | Güven:{p['conf']:.2f}")
    for rec in reversed(session_log):
        base = f"{time.strftime('%H:%M:%S', time.localtime(rec['when']))} | {rec['dealer']} | Son:{rec['last']} | Tahmin:{rec['adj_pred']} | Güven:{rec['conf']:.2f}"
        line = base + (" | WIN" if rec["result"] == "WIN" else f" | LOSE ({rec.get('actual')})")
        lines.append(line)
    path = os.path.join(APP_DIR, "log.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return send_file(path, as_attachment=True, download_name="log.txt")

if __name__ == "__main__":
    # Lokal geliştirme için:
    # python gork_web.py
    # Ardından http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
