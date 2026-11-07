"""
Meta-Cognitive Monitoring and Self-Correction System for EOS LLM.

Provides runtime meta-cognitive awareness, uncertainty quantification, Expected
Calibration Error (ECE) monitoring, conformal and free-energy OOD flags, and
flag-based self-correction across System One and deliberative decision layers.

Honesty contract: monitoring never invents inputs. Probabilities are never
synthesised from a stated confidence, energies are computed only from explicit
scores or logits, unlabeled decisions are excluded from ECE, and no stated
confidence is rewritten. Missing inputs are reported (for example through
``probabilities_missing``) instead of being filled in.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rhen.config import EOSConfig
except ImportError:
    from config import EOSConfig


class ConformalPredictionEngine:
    """
    Finite-sample Split-Conformal Prediction Sets with guaranteed coverage (1 - alpha).
    Computes empirical non-conformity quantile and outputs prediction sets C(S).

    The coverage guarantee only holds after calibrate() has been given a non-empty
    list of non-conformity scores; ``calibrated`` is then True. Until that happens,
    ``quantile_threshold`` (1 - alpha) is only a fallback, and predict_set() reports
    ``calibrated: False`` so callers can tell a fallback threshold from a calibrated one.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.calibration_scores: List[float] = []
        self.quantile_threshold: float = 1.0 - self.alpha  # Fallback until calibrate() runs
        self.calibrated: bool = False

    def calibrate(self, non_conformity_scores: List[float], alpha: Optional[float] = None) -> float:
        """
        Calibrates quantile threshold:
        q_hat = Quantile(scores; ceil((n + 1) * (1 - alpha)) / n)
        where non-conformity score is E_i = 1.0 - p(y_i).
        Sets ``calibrated`` to True for a non-empty score list; an empty list does not
        change the calibration state (scores, threshold or ``calibrated``).
        """
        if alpha is not None:
            self.alpha = float(alpha)
        target_alpha = self.alpha

        if not non_conformity_scores:
            return self.quantile_threshold

        self.calibration_scores = [float(s) for s in non_conformity_scores]
        n = len(self.calibration_scores)
        sorted_scores = sorted(self.calibration_scores)

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha))
        k = math.ceil((n + 1) * (1.0 - target_alpha))
        if k > n:
            q_hat = 1.0
        elif k <= 0:
            q_hat = sorted_scores[0]
        else:
            q_hat = sorted_scores[k - 1]

        self.quantile_threshold = float(q_hat)
        self.calibrated = True
        return self.quantile_threshold

    def predict_set(
        self,
        probabilities: Union[List[float], Dict[str, float]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns conformal prediction set C(S) = {o_k : p(o_k) >= 1.0 - q_hat}.
        Returns dict with:
        - prediction_set: list of option keys/indices in the set (always a list)
        - set_size: |C(S)|
        - is_singleton: bool (|C(S)| == 1)
        - is_ambiguous: bool (|C(S)| > 1)
        - is_empty: bool (|C(S)| == 0, indicates extreme OOD)
        - coverage_guarantee: 1.0 - alpha (nominal; it only holds when calibrated is True)
        - threshold: 1.0 - q_hat
        - calibrated: bool, True only after calibrate() with a non-empty score list.
          When False, the threshold is an uncalibrated fallback, not a guarantee.
        """
        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()  # arrays and tensors -> plain Python values

        target_alpha = float(alpha) if alpha is not None else self.alpha

        if alpha is not None and self.calibration_scores:
            n = len(self.calibration_scores)
            sorted_scores = sorted(self.calibration_scores)
            k = math.ceil((n + 1) * (1.0 - target_alpha))
            if k > n:
                q_hat = 1.0
            elif k <= 0:
                q_hat = sorted_scores[0]
            else:
                q_hat = sorted_scores[k - 1]
        elif alpha is not None:
            q_hat = 1.0 - target_alpha
        else:
            q_hat = self.quantile_threshold

        threshold = 1.0 - q_hat
        eps = 1e-9

        prediction_set: List[Any] = []
        if isinstance(probabilities, dict):
            for k_opt, p in probabilities.items():
                if float(p) >= threshold - eps:
                    prediction_set.append(k_opt)
        elif isinstance(probabilities, (list, tuple)):
            for idx, p in enumerate(probabilities):
                if float(p) >= threshold - eps:
                    prediction_set.append(idx)

        set_size = len(prediction_set)
        return {
            "prediction_set": prediction_set,
            "set_size": set_size,
            "is_singleton": set_size == 1,
            "is_ambiguous": set_size > 1,
            "is_empty": set_size == 0,
            "coverage_guarantee": 1.0 - target_alpha,
            "threshold": threshold,
            "calibrated": bool(self.calibrated),
        }


class AdaptiveConformalEngine:
    """
    Online Adaptive Conformal Risk Control via Robbins-Monro / Rolling Martingale.
    Tracks an empirical miss rate of alpha under non-stationary covariate shift
    (no finite-sample guarantee):
    q_{t+1} = clamp(q_t + eta * (error_t - alpha), 0.0, 1.0)
    where error_t = 1 if y_true not in C_t(S) else 0.

    Drift flag (``drift_detected``) is raised when either
    - there have been 3 or more consecutive misses, or
    - at least 10 updates are recorded and the rolling error over the previous
      (up to 20) updates exceeds 2.5 * alpha.
    The rolling window is taken before the current update is appended.
    """

    def __init__(self, alpha: float = 0.02, eta: float = 0.01, initial_quantile: Optional[float] = None):
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.quantile_threshold = float(initial_quantile) if initial_quantile is not None else (1.0 - self.alpha)
        self.history: List[Dict[str, Any]] = []
        self.consecutive_errors: int = 0
        self.drift_detected: bool = False

    def update(self, y_true: Any, prediction_set: List[Any], non_conformity_score: Optional[float] = None) -> Dict[str, Any]:
        # Check if y_true in prediction_set
        covered = (y_true in prediction_set)
        err = 0.0 if covered else 1.0
        if not covered:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        # Rolling martingale update: if error occurred, expand set by raising quantile threshold
        old_q = self.quantile_threshold
        self.quantile_threshold = max(0.0, min(1.0, self.quantile_threshold + self.eta * (err - self.alpha)))

        # Drift detection: 3 consecutive misses or rolling error > 2.5 * alpha (see class docstring)
        rolling_err = sum(1.0 for h in self.history[-20:] if not h["covered"]) / max(1, len(self.history[-20:])) if self.history else err
        self.drift_detected = (self.consecutive_errors >= 3) or (len(self.history) >= 10 and rolling_err > 2.5 * self.alpha)

        record = {
            "y_true": y_true,
            "covered": covered,
            "error": err,
            "old_quantile": old_q,
            "new_quantile": self.quantile_threshold,
            "threshold": 1.0 - self.quantile_threshold,
            "consecutive_errors": self.consecutive_errors,
            "rolling_error": rolling_err,
            "drift_detected": self.drift_detected,
        }
        if non_conformity_score is not None:
            record["non_conformity_score"] = float(non_conformity_score)
        self.history.append(record)
        return record

    def predict_set(self, probabilities: Union[List[float], Dict[str, float]]) -> Dict[str, Any]:
        threshold = max(0.0, 1.0 - self.quantile_threshold)
        eps = 1e-9

        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()

        prediction_set: List[Any] = []
        if isinstance(probabilities, dict):
            for k_opt, p in probabilities.items():
                if float(p) >= threshold - eps:
                    prediction_set.append(k_opt)
        elif isinstance(probabilities, (list, tuple)):
            for idx, p in enumerate(probabilities):
                if float(p) >= threshold - eps:
                    prediction_set.append(idx)

        set_size = len(prediction_set)
        return {
            "prediction_set": prediction_set,
            "set_size": set_size,
            "is_singleton": set_size == 1,
            "is_ambiguous": set_size > 1,
            "is_empty": set_size == 0,
            "threshold": threshold,
            "quantile_threshold": self.quantile_threshold,
            "drift_detected": self.drift_detected,
            "alpha": self.alpha,
            "coverage_guarantee": 1.0 - self.alpha,
        }


class EnergyOODDetector:
    """
    Free-energy based Out-Of-Distribution (OOD) detector.
    Computes Helmholtz free energy: E(q, S) = -tau * log sum_k exp(s_k / tau)
    Higher energy = unanswerable / out-of-distribution state.

    Inputs must be raw scores or logits, NOT probabilities. Probabilities are already
    normalised (softmax outputs), so a log-sum-exp over them is not the energy of the
    model's scores and the value is meaningless. Callers must pass explicit scores or
    logits, and skip the energy when only probabilities are available.
    """

    def __init__(self, ood_threshold: float = 5.0):
        self.ood_threshold = float(ood_threshold)

    def compute_energy(
        self,
        logits_or_scores: Union[List[float], Dict[str, float]],
        temperature: float = 1.0,
    ) -> float:
        """
        Computes -temperature * log(sum(exp(s_i / temperature))) over raw scores or
        logits (never probabilities). Returns inf for empty input.
        """
        if isinstance(logits_or_scores, dict):
            scores = [float(v) for v in logits_or_scores.values()]
        elif isinstance(logits_or_scores, (list, tuple)):
            scores = [float(v) for v in logits_or_scores]
        else:
            scores = []

        if not scores:
            return float("inf")

        tau = max(1e-6, float(temperature))
        scaled = [s / tau for s in scores]
        max_s = max(scaled)

        # Log-sum-exp trick for numerical stability
        sum_exp = sum(math.exp(s - max_s) for s in scaled)
        log_sum_exp = max_s + math.log(max(sum_exp, 1e-12))
        return float(-tau * log_sum_exp)

    def is_ood(self, energy: float, threshold_override: Optional[float] = None) -> bool:
        """Checks if computed Helmholtz free energy exceeds OOD threshold."""
        thresh = float(threshold_override) if threshold_override is not None else self.ood_threshold
        if math.isnan(energy):
            return True
        return bool(energy >= thresh)


class MetaCognitiveMonitor:
    """
    Meta-cognitive monitoring and self-correction system.
    
    Functions:
    1. Tracks decision history and confidence distributions across bins.
    2. Measures Expected Calibration Error (ECE) against empirical correctness.
    3. Quantifies Epistemic vs. Aleatoric uncertainty.
    4. Provides Platt-style scaling as a standalone utility (apply_platt_scaling). Decision
       monitoring never rewrites a decision's confidence; it attaches flags only.
    5. Detects anomaly patterns and flags reasoning escalation when warranted.
    """
    
    def __init__(self, config: EOSConfig):
        self.config = config
        self.decision_history: List[Dict[str, Any]] = []
        # Per-mode calibration factors. Nothing writes them, so they stay empty and no
        # confidence is adjusted from them; kept so get_calibration_report() is stable.
        self.confidence_calibration: Dict[str, float] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.success_patterns: Dict[str, int] = defaultdict(int)
        
        # Binned calibration parameters (10 equal bins in [0, 1])
        self.calibration_history: List[Dict[str, Any]] = []
        self.confidence_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.bin_predictions: Dict[int, List[float]] = {i: [] for i in range(len(self.confidence_bins) - 1)}
        self.bin_accuracies: Dict[int, List[float]] = {i: [] for i in range(len(self.confidence_bins) - 1)}
        
        # Platt scaling parameters: calibrated_p = 1 / (1 + exp(A * p + B)).
        # Reference values only: decision monitoring does not apply them.
        self.platt_a = -1.0
        self.platt_b = 0.0

        # Conformal Prediction and Energy OOD Detectors
        self.conformal_engine = ConformalPredictionEngine(alpha=0.05)
        self.adaptive_conformal = AdaptiveConformalEngine(alpha=0.02)
        self.energy_detector = EnergyOODDetector(ood_threshold=4.5)

    # --------------------------------------------------------------------------
    # Uncertainty Decomposition
    # --------------------------------------------------------------------------

    def compute_uncertainty(self, probabilities: Union[List[float], Dict[str, float]]) -> Dict[str, float]:
        """
        Decomposes total uncertainty into Epistemic and Aleatoric components.
        
        - Aleatoric Uncertainty: Shannon entropy of the decision distribution H(p).
        - Epistemic Uncertainty: Representation divergence / variance (estimated from confidence gap).
        - Total Uncertainty: Normalized combination in [0, 1].
        """
        if isinstance(probabilities, dict):
            probs = list(probabilities.values())
        else:
            probs = list(probabilities)
        
        if not probs:
            return {"aleatoric": 1.0, "epistemic": 1.0, "total_uncertainty": 1.0}

        # Normalize probabilities
        total_p = sum(probs)
        if total_p <= 0:
            probs = [1.0 / len(probs)] * len(probs)
        else:
            probs = [p / total_p for p in probs]

        # Shannon Entropy H(p) = - sum(p * log(p))
        k = len(probs)
        max_entropy = math.log(k) if k > 1 else 1.0
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        aleatoric = min(1.0, max(0.0, entropy / max(max_entropy, 1e-6)))

        # Epistemic uncertainty: derived from dispersion among top-2 hypotheses
        sorted_probs = sorted(probs, reverse=True)
        top1 = sorted_probs[0] if len(sorted_probs) > 0 else 1.0
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2
        epistemic = max(0.0, min(1.0, 1.0 - margin))

        total_uncertainty = 0.5 * aleatoric + 0.5 * epistemic
        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
        }

    def decompose_uncertainty(self, probabilities: Union[List[float], Dict[str, float]]) -> Dict[str, float]:
        """Convenience alias for compute_uncertainty."""
        return self.compute_uncertainty(probabilities)

    # --------------------------------------------------------------------------
    # Expected Calibration Error (ECE)
    # --------------------------------------------------------------------------

    def compute_ece(
        self,
        decisions: Optional[List[Dict[str, Any]]] = None,
        n_bins: int = 10,
    ) -> float:
        """
        Computes empirical Expected Calibration Error (ECE):
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

        With ``decisions``, only entries that carry a ``ground_truth`` label (not None)
        and a ``confidence`` are counted; the others are skipped rather than assumed
        correct or given a default confidence. Returns 0.0 when no entry is usable.
        Without ``decisions``, the recorded calibration bins are used.
        """
        if decisions is not None:
            # Only labeled decisions with a stated confidence are counted.
            usable: List[Tuple[float, float]] = []
            for d in decisions:
                if not isinstance(d, dict) or d.get("ground_truth") is None or d.get("confidence") is None:
                    continue
                conf = float(d["confidence"])
                if not math.isfinite(conf):
                    continue
                usable.append((min(1.0, max(0.0, conf)), float(d["ground_truth"])))
            if not usable:
                return 0.0
            
            # Custom binned computation on provided decisions
            bins_conf = [[] for _ in range(n_bins)]
            bins_acc = [[] for _ in range(n_bins)]

            for conf, gt in usable:
                # Assign to bin
                bin_idx = min(int(conf * n_bins), n_bins - 1)
                bins_conf[bin_idx].append(conf)
                bins_acc[bin_idx].append(gt)

            total_n = len(usable)
            ece = 0.0
            for b_idx in range(n_bins):
                confs = bins_conf[b_idx]
                accs = bins_acc[b_idx]
                if confs:
                    b_conf = sum(confs) / len(confs)
                    b_acc = sum(accs) / len(accs)
                    weight = len(confs) / total_n
                    ece += weight * abs(b_acc - b_conf)
            return ece

        total_samples = sum(len(preds) for preds in self.bin_predictions.values())
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for i in range(len(self.confidence_bins) - 1):
            preds = self.bin_predictions[i]
            accs = self.bin_accuracies[i]
            if len(preds) > 0 and len(accs) > 0:
                bin_conf = sum(preds) / len(preds)
                bin_acc = sum(accs) / len(accs)
                weight = len(preds) / total_samples
                ece += weight * abs(bin_acc - bin_conf)

        return ece

    def apply_platt_scaling(
        self,
        confidence: float,
        temperature: float = 1.0,
        a: float = 1.0,
        b: float = 0.0,
    ) -> float:
        """
        Applies Platt scaling and temperature adjustment to confidence score.
        p_scaled = sigmoid((logit(p) * a + b) / temperature)
        """
        p = max(1e-6, min(1.0 - 1e-6, float(confidence)))
        logit = math.log(p / (1.0 - p))
        scaled_logit = (logit * a + b) / max(0.01, temperature)
        scaled_conf = 1.0 / (1.0 + math.exp(-scaled_logit))
        return max(0.001, min(0.999, scaled_conf))

    def detect_confidence_anomaly(
        self,
        confidence: float,
        probabilities: Union[List[float], Dict[str, float]],
    ) -> bool:
        """
        Detects anomalies such as high-entropy decision distributions
        accompanied by high stated confidence scores.
        """
        unc = self.compute_uncertainty(probabilities)
        aleatoric = unc.get("aleatoric", 0.0)
        # Flag anomaly if distribution has high entropy (> 0.7) but confidence is high (> 0.8)
        if aleatoric > 0.70 and confidence > 0.80:
            return True
        return False

    # --------------------------------------------------------------------------
    # Conformal Prediction & Energy OOD Detection
    # --------------------------------------------------------------------------

    def compute_conformal_set(
        self,
        probabilities: Union[List[float], Dict[str, float]],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Computes finite-sample conformal prediction set C(S) with guaranteed coverage (1 - alpha).
        The guarantee only holds once the engine is calibrated; the result's ``calibrated``
        key reports this.
        """
        return self.conformal_engine.predict_set(probabilities, alpha=alpha)

    def compute_adaptive_conformal_set(
        self,
        probabilities: Union[List[float], Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Computes online adaptive conformal prediction set C_t(S) with empirical coverage guarantee.
        """
        return self.adaptive_conformal.predict_set(probabilities)

    def update_adaptive_conformal(
        self,
        y_true: Any,
        prediction_set: Optional[List[Any]] = None,
        probabilities: Optional[Union[List[float], Dict[str, float]]] = None,
        non_conformity_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates the online adaptive conformal engine with observed ground truth.
        If prediction_set is omitted, it is dynamically computed from probabilities.
        """
        if prediction_set is None:
            if probabilities is not None:
                prediction_set = self.adaptive_conformal.predict_set(probabilities)["prediction_set"]
            else:
                prediction_set = []
        return self.adaptive_conformal.update(
            y_true=y_true,
            prediction_set=prediction_set,
            non_conformity_score=non_conformity_score,
        )

    def compute_free_energy(
        self,
        scores: Union[List[float], Dict[str, float]],
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Computes Helmholtz free energy E(q, S) and evaluates OOD status.
        Higher energy indicates unanswerable / out-of-distribution state.
        ``scores`` must be raw scores or logits, never probabilities.
        """
        energy = self.energy_detector.compute_energy(scores, temperature=temperature)
        is_ood = self.energy_detector.is_ood(energy)
        return {
            "energy": energy,
            "is_ood": is_ood,
            "temperature": temperature,
            "ood_threshold": self.energy_detector.ood_threshold,
        }

    # --------------------------------------------------------------------------
    # Decision Monitoring & Self-Correction (Async + Sync)
    # --------------------------------------------------------------------------

    async def monitor_decision(
        self,
        decision_result: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """Async monitor decision quality and apply meta-cognitive corrections."""
        return self.monitor_decision_sync(decision_result, context)

    def monitor_decision_sync(
        self,
        decision_result: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """
        Synchronously monitor decision quality and attach diagnostics and flags.

        Only explicit inputs are used. The probability vector is never synthesised from
        the stated confidence: when none is present, ``probabilities_missing`` is set and
        no conformal set or energy is computed. Energy is computed only from explicit
        scores or logits, never from probabilities. The stated confidence is never rewritten.
        """
        confidence = self._extract_confidence(decision_result)
        result_type = decision_result.get("mode", "unknown")
        
        # Explicit inputs only. Aliases are accepted, but nothing is synthesised.
        raw_probs = self._explicit_vector(decision_result.get("probabilities"))
        if raw_probs is None:
            raw_probs = self._explicit_vector(decision_result.get("raw_probabilities"))
        raw_scores = self._explicit_vector(decision_result.get("scores"))
        if raw_scores is None:
            raw_scores = self._explicit_vector(decision_result.get("logits"))

        if raw_probs is None and raw_scores is None and isinstance(decision_result.get("results"), list) and decision_result["results"]:
            first_res = decision_result["results"][0]
            if isinstance(first_res, dict):
                raw_probs = self._explicit_vector(first_res.get("probabilities"))
                if raw_probs is None:
                    raw_probs = self._explicit_vector(first_res.get("raw_probabilities"))
                raw_scores = self._explicit_vector(first_res.get("scores"))
                if raw_scores is None:
                    raw_scores = self._explicit_vector(first_res.get("logits"))
            elif hasattr(first_res, "probabilities"):
                raw_probs = self._explicit_vector(first_res.probabilities)

        # Uncertainty comes only from an explicit probability vector. Without one, no
        # distribution is assumed and no uncertainty estimate is reported.
        if raw_probs is not None:
            uncertainty = self.compute_uncertainty(raw_probs)
        else:
            uncertainty = {}
        decision_result["uncertainty"] = uncertainty

        conformal_set: Dict[str, Any] = {}
        adaptive_conformal_set: Dict[str, Any] = {}
        if raw_probs is None:
            # Conformal sets need an explicit probability vector; none is computed without one.
            decision_result["probabilities_missing"] = True
        else:
            conformal_set = self.compute_conformal_set(raw_probs)
            adaptive_conformal_set = self.compute_adaptive_conformal_set(raw_probs)
            decision_result["conformal_set"] = conformal_set
            decision_result["adaptive_conformal_set"] = adaptive_conformal_set

        # Energy uses explicit scores or logits only, whether or not probabilities are present.
        # Probabilities are never used as scores.
        energy_ood: Dict[str, Any] = {}
        if raw_scores is not None:
            energy_ood = self.compute_free_energy(raw_scores)
            decision_result["energy_ood"] = energy_ood

        # A conformal ambiguity is trusted only from a calibrated engine.
        conformal_ambiguous = bool(conformal_set.get("is_ambiguous") and conformal_set.get("calibrated"))

        # Flag in meta_cognitive_correction to recommend deliberative escalation
        if conformal_ambiguous or energy_ood.get("is_ood") or adaptive_conformal_set.get("drift_detected"):
            if "meta_cognitive_correction" not in decision_result or not isinstance(decision_result["meta_cognitive_correction"], dict):
                decision_result["meta_cognitive_correction"] = {}
            decision_result["meta_cognitive_correction"]["recommend_escalation"] = True
            decision_result["meta_cognitive_correction"]["recommendation"] = "deliberative_escalation"
            decision_result["meta_cognitive_correction"]["deliberative_escalation"] = True
            if conformal_ambiguous:
                decision_result["meta_cognitive_correction"]["conformal_ambiguity"] = True
            if adaptive_conformal_set.get("drift_detected"):
                decision_result["meta_cognitive_correction"]["adaptive_conformal_drift"] = True
            if energy_ood.get("is_ood"):
                decision_result["meta_cognitive_correction"]["energy_ood"] = True

        # Record decision in history
        self.decision_history.append({
            "result_type": result_type,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "layer_path": self._extract_path(decision_result),
        })

        # Apply self-correction if enabled and sufficient history exists
        if self.config.rhen_enable_meta_cognitive and len(self.decision_history) >= 2:
            correction = self.self_correct_sync(decision_result, context)
            if correction.get("applied"):
                if "meta_cognitive_correction" not in decision_result or not isinstance(decision_result["meta_cognitive_correction"], dict):
                    decision_result["meta_cognitive_correction"] = {}
                decision_result["meta_cognitive_correction"].update(correction["corrections"])

        return decision_result

    async def self_correct(
        self,
        initial_decision: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """Async self-correction."""
        return self.self_correct_sync(initial_decision, context)

    def self_correct_sync(
        self,
        initial_decision: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """
        Applies self-correction based on meta-cognitive analysis.

        Only flags are attached (anomaly_detected, conformal_ambiguity, energy_ood and the
        escalation fields). The initial decision is not modified, and its stated confidence
        is never rewritten or rescaled.
        """
        corrections: Dict[str, Any] = {}
        applied = False

        confidence = self._extract_confidence(initial_decision)
        result_type = initial_decision.get("mode", "unknown")

        # 1. Anomaly detection: high aleatoric uncertainty with high stated confidence
        uncertainty = initial_decision.get("uncertainty", {})
        if uncertainty.get("aleatoric", 0.0) > 0.7 and confidence > 0.8:
            corrections["anomaly_detected"] = "high_entropy_high_confidence_conflict"
            self.error_patterns["entropy_conflict"] += 1
            applied = True

        # 2. Conformal prediction ambiguity or Energy OOD escalation
        conformal_set = initial_decision.get("conformal_set")
        energy_ood = initial_decision.get("energy_ood")
        if conformal_set and conformal_set.get("is_ambiguous") and conformal_set.get("calibrated"):
            corrections["conformal_ambiguity"] = True
            corrections["recommend_escalation"] = True
            corrections["recommendation"] = "deliberative_escalation"
            corrections["deliberative_escalation"] = True
            applied = True
        if energy_ood and energy_ood.get("is_ood"):
            corrections["energy_ood"] = True
            corrections["recommend_escalation"] = True
            corrections["recommendation"] = "deliberative_escalation"
            corrections["deliberative_escalation"] = True
            applied = True

        if applied:
            self.error_patterns[result_type] += 1
        else:
            self.success_patterns[result_type] += 1

        return {"applied": applied, "corrections": corrections}

    # --------------------------------------------------------------------------
    # Batch Quality Analysis & Diagnostics
    # --------------------------------------------------------------------------

    async def analyze_decision_quality(
        self,
        results: List[Any],
        context: Any,
    ) -> Dict[str, Any]:
        """Async analysis wrapper."""
        return self.analyze_decision_quality_sync(results, context)

    def analyze_decision_quality_sync(
        self,
        results: List[Any],
        context: Any,
    ) -> Dict[str, Any]:
        """
        Analyzes quality across a batch of decisions (distribution and recommendations).

        This is read-only with respect to calibration: the batch is not added to the ECE
        bins, so compute_ece() is unchanged by this call.
        """
        analysis: Dict[str, Any] = {
            "total_decisions": len(results),
            "avg_confidence": 0.0,
            "ece": self.compute_ece(),
            "confidence_distribution": {},
            "recommendations": [],
        }
        
        if not results:
            return analysis

        confidences = [self._extract_confidence(r) for r in results]
        avg_conf = sum(confidences) / len(confidences)
        analysis["avg_confidence"] = avg_conf

        # Populate binned distribution
        bin_counts = defaultdict(int)
        for conf in confidences:
            bin_idx = min(len(self.confidence_bins) - 2, int(conf * (len(self.confidence_bins) - 1)))
            bin_counts[bin_idx] += 1

        for i in range(len(self.confidence_bins) - 1):
            low, high = self.confidence_bins[i], self.confidence_bins[i + 1]
            cnt = bin_counts[i]
            if cnt > 0:
                analysis["confidence_distribution"][f"{low:.1f}-{high:.1f}"] = {
                    "count": cnt,
                    "percentage": (cnt / len(results)) * 100.0,
                }

        # Recommendations based on empirical health
        if avg_conf < 0.60:
            analysis["recommendations"].append("High aggregate uncertainty: recommend routing low-confidence queries to Reasoning Layer.")
        if self.compute_ece() > 0.15:
            analysis["recommendations"].append("Expected Calibration Error > 15%: recommend updating Platt scaling factors.")

        return analysis

    # --------------------------------------------------------------------------
    # Supervised / Feedback Calibration
    # --------------------------------------------------------------------------

    async def update_calibration_from_feedback(
        self,
        decision_id: str,
        actual_correctness: bool,
        predicted_confidence: float,
    ) -> None:
        """Async feedback calibration wrapper."""
        self.update_calibration_from_feedback_sync(decision_id, actual_correctness, predicted_confidence)

    def update_calibration_from_feedback_sync(
        self,
        decision_id: str,
        actual_correctness: bool,
        predicted_confidence: float,
    ) -> None:
        """
        Records ground-truth correctness feedback in the calibration bins and the
        calibration history (the bins feed compute_ece). No confidence or scaling
        parameter is adjusted from feedback.
        """
        conf = max(0.0, min(1.0, float(predicted_confidence)))
        acc = 1.0 if actual_correctness else 0.0
        
        bin_idx = min(len(self.confidence_bins) - 2, int(conf * (len(self.confidence_bins) - 1)))
        self.bin_predictions[bin_idx].append(conf)
        self.bin_accuracies[bin_idx].append(acc)

        self.calibration_history.append({
            "id": decision_id,
            "confidence": conf,
            "correctness": acc,
        })

    @staticmethod
    def _explicit_vector(value: Any) -> Optional[Union[List[float], Dict[str, float]]]:
        """
        Returns a plain copy of an explicitly supplied probability or score vector, or
        None when the value is absent, empty or not a vector. Tensors and arrays are
        converted with tolist(). Nothing is synthesised.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value) if value else None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value) if value else None
        return None

    def _extract_confidence(self, result: Any) -> float:
        """Extracts normalized confidence from diverse result shapes."""
        if isinstance(result, dict):
            return float(result.get("confidence", 0.5))
        elif hasattr(result, "confidence"):
            return float(result.confidence)
        return 0.5

    def _extract_path(self, result: Dict[str, Any]) -> List[str]:
        """Extracts execution layer path."""
        if isinstance(result, dict):
            return result.get("path_taken", ["semantic"])
        return ["semantic"]

    def get_calibration_report(self) -> Dict[str, Any]:
        """Returns a comprehensive calibration and health diagnostic report."""
        return {
            "total_decisions_monitored": len(self.decision_history),
            "total_feedback_samples": len(self.calibration_history),
            "expected_calibration_error": self.compute_ece(),
            "calibration_factors": self.confidence_calibration.copy(),
            "error_patterns": dict(self.error_patterns),
            "success_patterns": dict(self.success_patterns),
        }


class RhenCalibrator:
    """
    Rhen Post-Hoc Calibrator (Jev & Kev alignment engine).
    
    Provides:
    1. Golden-Section search temperature fitting over log grid [0.25, 4.0] minimizing NLL.
    2. Selective classification / Coverage-at-5%-error metric evaluation.
    3. Proper Brier score computation.
    """

    @staticmethod
    def fit_temperature(
        logits: List[List[float]],
        labels: List[int],
        bounds: Tuple[float, float] = (0.25, 4.0),
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> float:
        """
        Fits optimal post-hoc scaling temperature T* using Golden-Section search
        in log-temperature space to minimize validation Cross-Entropy NLL.
        """
        if not logits or not labels or len(logits) != len(labels):
            return 1.0

        def nll_for_temp(T: float) -> float:
            total_nll = 0.0
            for row, target in zip(logits, labels):
                # Scaled logits
                scaled = [z / max(T, 1e-6) for z in row]
                max_z = max(scaled)
                exp_z = [math.exp(z - max_z) for z in scaled]
                sum_exp = sum(exp_z)
                target_p = exp_z[target] / max(sum_exp, 1e-12)
                total_nll -= math.log(max(target_p, 1e-12))
            return total_nll / len(logits)

        # Golden-section search in log-space
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        resphi = 2.0 - phi

        a = math.log(bounds[0])
        b = math.log(bounds[1])

        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1 = nll_for_temp(math.exp(x1))
        f2 = nll_for_temp(math.exp(x2))

        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = nll_for_temp(math.exp(x1))
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2 = nll_for_temp(math.exp(x2))

        best_log_t = (a + b) / 2.0
        return float(math.exp(best_log_t))

    @staticmethod
    def compute_brier_score(probabilities: List[List[float]], labels: List[int]) -> float:
        """
        Computes multi-class Brier score: (1/N) * sum_{i=1}^N (1/K) * sum_{k=1}^K (p_ik - y_ik)^2.
        """
        if not probabilities or not labels or len(probabilities) != len(labels):
            return 0.0
        total_brier = 0.0
        for probs, target in zip(probabilities, labels):
            K = len(probs)
            row_brier = sum((p - (1.0 if k == target else 0.0)) ** 2 for k, p in enumerate(probs)) / max(K, 1)
            total_brier += row_brier
        return total_brier / len(probabilities)

    @staticmethod
    def compute_coverage_at_error_budget(
        probabilities: List[List[float]],
        labels: List[int],
        error_budget: float = 0.05,
    ) -> Dict[str, float]:
        """
        Evaluates selective classification: accepts predictions with confidence >= threshold
        guaranteeing the empirical error on accepted samples does not exceed error_budget.
        """
        if not probabilities or not labels or len(probabilities) != len(labels):
            return {"coverage": 0.0, "threshold": 1.0, "accepted_error": 0.0, "accepted_count": 0}

        samples = []
        for i, (probs, target) in enumerate(zip(probabilities, labels)):
            pred_idx = max(range(len(probs)), key=lambda k: probs[k])
            conf = probs[pred_idx]
            is_correct = (pred_idx == target)
            samples.append((conf, is_correct))

        # Sort descending by confidence
        samples.sort(key=lambda s: s[0], reverse=True)

        accepted_errors = 0
        best_threshold = 1.0
        best_count = 0

        for i, (conf, correct) in enumerate(samples):
            current_errors = accepted_errors + (0 if correct else 1)
            current_count = i + 1
            if current_errors / current_count <= error_budget:
                accepted_errors = current_errors
                best_threshold = conf
                best_count = current_count

        total_n = len(samples)
        coverage = best_count / total_n if total_n > 0 else 0.0
        accepted_error = (accepted_errors / best_count) if best_count > 0 else 0.0

        return {
            "coverage": coverage,
            "threshold": float(best_threshold),
            "accepted_error": accepted_error,
            "accepted_count": best_count,
            "total_count": total_n,
        }