"""
Dual-AI Decision Engine: Grok Forecaster + GPT Critic

This engine orchestrates a two-stage AI decision process:
1. Grok (xAI) acts as Forecaster - researches and predicts outcomes
2. GPT (OpenAI) acts as Critic - validates forecasts before trading
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.utils.logging_setup import TradingLoggerMixin, get_trading_logger
from src.config.settings import settings


@dataclass
class ForecastResult:
    """Grok's forecast with evidence."""
    market_id: str
    predicted_probability: float
    confidence: float
    side: str  # "YES" or "NO"
    action: str  # "BUY" or "SKIP"
    evidence: List[str]
    key_factors: List[str]
    risks: List[str]
    reasoning: str
    forecast_cost: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CriticReview:
    """GPT's review of the forecast."""
    approved: bool
    agreement_score: float  # 0.0 to 1.0
    critique: str
    suggested_adjustments: Optional[Dict[str, Any]] = None
    final_recommendation: str = "REJECT"  # "APPROVE", "REJECT", "MODIFY"
    rejection_reason: Optional[str] = None
    review_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DualAIDecision:
    """Final combined decision from both AIs."""
    forecast: ForecastResult
    review: CriticReview
    final_action: str  # "BUY" or "SKIP"
    final_side: str  # "YES" or "NO"
    final_confidence: float
    total_cost: float
    dual_ai_reasoning: str


# Prompt templates for the dual-AI system
GROK_FORECASTER_PROMPT = """
You are an expert prediction market researcher and forecaster. Your job is to conduct rigorous, structured analysis before any trading recommendation.

---
## MARKET CONTEXT

**Title:** {title}
**Rules:** {rules}
**Current YES Price:** {yes_price}¢  |  **Current NO Price:** {no_price}¢
**Volume:** ${volume:,.0f}
**Expires In:** {days_to_expiry:.1f} days ({hours_to_expiry:.0f} hours)

**Live Research Context:**
{news_summary}

**Portfolio:**
- Available Cash: ${cash:,.2f}
- Max Trade Value: ${max_trade_value:,.2f}

---
## PHASE 1: RESOLUTION CRITERIA ANALYSIS

Before estimating probability, you MUST answer:
1. What EXACTLY triggers YES vs NO resolution?
2. What is the MINIMUM threshold for YES?
3. Are there edge cases, ambiguities, or "technicality" outcomes?
4. Can the resolution be gamed or manipulated?
5. What is the "minimum compliance" or "lazy/obvious" outcome?

---
## PHASE 2: BASE RATE ANALYSIS

1. What has happened in similar historical situations?
2. What is the prior probability BEFORE considering this specific case?
3. How does this case differ from the base rate?
4. Sample size: How many comparable cases exist?
5. **Red flag check:** Am I making a "this time is different" argument?

---
## PHASE 3: INCENTIVE ANALYSIS

1. Who are the key actors/decision-makers?
2. What are their incentives (political, financial, reputational)?
3. What is the path of least resistance for them?
4. What would be surprising vs expected behavior?
5. Do they face different consequences for action vs inaction?

---
## PHASE 4: PROBABILITY ESTIMATION

Based on your analysis above:
1. **Your TRUE probability estimate:** [X]% (be precise)
2. **Confidence interval:** [Low]% to [High]% (90% CI)
3. **Key assumptions:** List 3-5 critical assumptions
4. **Swing factors:** Which factors could flip the outcome?

---
## PHASE 5: EDGE ARTICULATION

**In ONE sentence, what is your edge?**
"My edge is: ________________________________"

CRITICAL: If you cannot articulate a specific edge beyond "I disagree with the market," the answer is SKIP.

---
## TRADING RECOMMENDATION

Only recommend BUY if:
- Your probability differs from market price by >10%
- You can articulate a clear, specific edge
- Confidence is >60%

---
## REQUIRED OUTPUT (JSON only, no other text)

```json
{{
"predicted_probability": float (0.0-1.0, your true probability estimate),
"confidence": float (0.0-1.0, how confident you are in this estimate),
"confidence_interval_low": float (lower bound of 90% CI),
"confidence_interval_high": float (upper bound of 90% CI),
"side": "YES" | "NO",
"action": "BUY" | "SKIP",
"edge_statement": "One sentence articulating your specific edge",
"evidence": ["Evidence point 1", "Evidence point 2", "..."],
"key_factors": ["Factor 1 with weight", "Factor 2 with weight", "..."],
"risks": ["Risk 1", "Risk 2", "..."],
"base_rate": "Description of historical base rate and how this case differs",
"resolution_clarity": "How clear is the resolution criteria? High/Medium/Low",
"reasoning": "Detailed explanation covering all phases of analysis"
}}
```
"""

GPT_CRITIC_PROMPT = """
You are a trading risk manager and CRITIC. Your role is to rigorously challenge the forecaster's analysis and protect against bad trades.

A forecaster has analyzed a prediction market and recommends a trade. Your job is to:
1. Critically review the analysis for flaws and biases
2. Steel-man the opposite position
3. Verify the claimed edge is real and specific
4. Decide whether to APPROVE or REJECT

---
## MARKET CONTEXT

**Title:** {title}
**Current YES Price:** {yes_price}¢  |  **Current NO Price:** {no_price}¢
**Volume:** ${volume:,.0f}
**Expires:** {days_to_expiry:.1f} days ({hours_to_expiry:.0f} hours)

---
## FORECASTER'S ANALYSIS

**Predicted Probability:** {predicted_probability:.1%}
**Confidence:** {confidence:.1%}
**Recommended Side:** {side}
**Recommended Action:** {action}

**Evidence Provided:**
{evidence}

**Key Factors:**
{key_factors}

**Identified Risks:**
{risks}

**Forecaster's Reasoning:**
{reasoning}

---
## YOUR CRITICAL REVIEW

### Section 1: Cognitive Bias Check

Evaluate the forecaster's analysis for these biases:

☐ **Confirmation Bias:** Is the forecaster cherry-picking evidence that supports their view?
☐ **Recency Bias:** Are they overweighting recent events vs historical patterns?
☐ **Availability Bias:** Are they overweighting vivid/memorable examples?
☐ **Overconfidence:** Is the confidence level appropriate given the uncertainty?
☐ **Anchoring:** Are they anchored to the current market price or initial estimate?

**Bias assessment:** [None detected / Minor concerns / Significant red flags]

### Section 2: Steel-Man the Opposite Position

If the forecaster recommends YES, make the BEST CASE for NO.
If the forecaster recommends NO, make the BEST CASE for YES.
If they recommend SKIP, is there actually a trade here they're missing?

**Strongest argument for the opposite side:**
[Your steel-man argument]

**How strong is this counter-argument?** [Weak / Moderate / Strong / Compelling]

### Section 3: Edge Verification

Critically examine the claimed edge:

1. **Is the edge clearly articulated?** The forecaster should have a one-sentence edge statement.
2. **Is this NOVEL analysis or just disagreeing with consensus?**
    - Novel analysis = specific insight others miss
    - Just disagreeing = dangerous overconfidence
3. **Do they have information the market doesn't?** (Rarely true)
4. **Do they have better analysis/modeling?** (Possible)
5. **Are they seeing structural factors others miss?** (Possible)

**Edge assessment:** [Real and specific / Vague / Non-existent]

### Section 4: Execution Feasibility

1. **Liquidity:** Can you actually get filled at these odds?
2. **Time Value:** Is locking up capital for {hours_to_expiry:.0f} hours worth it?
3. **Risk/Reward:** Does the potential gain justify the risk?

### Section 5: Resolution Clarity Check

1. Did the forecaster clearly understand what triggers YES vs NO?
2. Are there edge cases or ambiguities they missed?
3. Could this resolve on a technicality they didn't consider?

---
## DECISION RULES

**APPROVE if:**
- Analysis is rigorous and covers all phases
- Cognitive bias check passed
- Edge is clearly articulated and defensible
- Steel-man of opposite position was considered and addressed
- You genuinely believe the forecaster is likely correct

**REJECT if:**
- Significant cognitive biases detected that invalidate the thesis
- Edge is non-existent AND the opportunity is negative EV
- Steel-man argument for opposite side is CLEARLY SUPERIOR and refutes the thesis
- PROPER REJECTION: Don't reject just for minor formatting or phrasing. If the trade is positive EV but the edge is poorly phrased, APPROVE and suggest better phrasing.

**If the forecaster recommends SKIP:** Usually agree unless you see a clear opportunity they missed.

---
## REQUIRED OUTPUT (JSON only, no other text)

```json
{{
"approved": boolean,
"agreement_score": float (0.0-1.0, how much you agree with the forecaster),
"bias_assessment": "None detected" | "Minor concerns" | "Significant red flags",
"steel_man_strength": "Weak" | "Moderate" | "Strong" | "Compelling",
"edge_assessment": "Real and specific" | "Vague" | "Non-existent",
"critique": "Your detailed critical analysis of the forecast",
"steel_man_argument": "Your best argument for the opposite position",
"suggested_adjustments": {{
    "adjusted_probability": float (0.0-1.0),
    "adjusted_confidence": float (0.0-1.0)
}},
"final_recommendation": "APPROVE" | "REJECT",
"rejection_reason": "Specific reason for rejection, or null if approved"
}}
```
"""


class DualAIDecisionEngine(TradingLoggerMixin):
    """
    Orchestrates Grok (forecaster) + GPT (critic) dual-AI trading decisions.

    The dual-AI system ensures higher quality trades by:
    1. Having Grok research and forecast with evidence
    2. Having GPT critically review before trade execution
    3. Only executing trades that both AIs agree on
    """

    def __init__(self, xai_client=None, openai_client=None, db_manager=None, kalshi_client=None):
        """
        Initialize the dual-AI decision engine.

        Args:
            xai_client: XAIClient for Grok forecasting
            openai_client: OpenAIClient for GPT critic review
            db_manager: Optional DatabaseManager for logging
            kalshi_client: Optional KalshiClient for ML predictions
        """
        self.xai_client = xai_client
        self.openai_client = openai_client
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client

        # If clients not provided, create them lazily
        self._xai_initialized = xai_client is not None
        self._openai_initialized = openai_client is not None

        self.logger.info("DualAIDecisionEngine initialized")

    async def _ensure_clients(self):
        """Lazily initialize clients if not provided."""
        if not self._xai_initialized:
            from src.clients.xai_client import XAIClient
            self.xai_client = XAIClient(db_manager=self.db_manager, kalshi_client=self.kalshi_client)
            self._xai_initialized = True

        if not self._openai_initialized:
            from src.clients.openai_client import OpenAIClient
            self.openai_client = OpenAIClient()
            self._openai_initialized = True

    async def get_dual_ai_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str = ""
    ) -> Optional[DualAIDecision]:
        """
        Get a trading decision using the dual-AI (Grok forecaster + GPT critic) system.

        Args:
            market_data: Market information (title, prices, volume, etc.)
            portfolio_data: Portfolio state (balance, positions, etc.)
            news_summary: Optional news/research context

        Returns:
            DualAIDecision if both AIs reach a conclusion, None on error
        """
        await self._ensure_clients()

        market_id = market_data.get('ticker', market_data.get('market_id', 'unknown'))

        self.logger.info(
            f"🔮 [DUAL-AI] Starting analysis for {market_id}",
            market_title=market_data.get('title', '')[:50]
        )

        try:
            # Stage 1: Grok Forecaster
            self.logger.info("📊 [FORECASTER] Grok analyzing market...")
            forecast = await self._get_grok_forecast(
                market_data, portfolio_data, news_summary
            )

            if not forecast:
                self.logger.warning(f"❌ [FORECASTER] Grok failed to generate forecast for {market_id}")
                return None

            self.logger.info(
                f"📊 [FORECASTER] Grok forecast: {forecast.action} {forecast.side} "
                f"(prob: {forecast.predicted_probability:.1%}, conf: {forecast.confidence:.1%})"
            )

            # If forecaster says SKIP, we can short-circuit
            if forecast.action == "SKIP":
                self.logger.info(f"⏭️ [DUAL-AI] Forecaster recommends SKIP, skipping critic review")

                # Create a lightweight critic review agreeing with SKIP
                skip_review = CriticReview(
                    approved=True,
                    agreement_score=1.0,
                    critique="Forecaster recommends no trade - agreed.",
                    final_recommendation="APPROVE",
                    review_cost=0.0
                )

                return DualAIDecision(
                    forecast=forecast,
                    review=skip_review,
                    final_action="SKIP",
                    final_side=forecast.side,
                    final_confidence=forecast.confidence,
                    total_cost=forecast.forecast_cost,
                    dual_ai_reasoning=f"[FORECASTER SKIP] {forecast.reasoning}"
                )

            # Stage 2: GPT Critic Review
            self.logger.info("🔍 [CRITIC] GPT reviewing forecast...")
            review = await self._get_gpt_review(market_data, forecast)

            if not review:
                self.logger.warning(f"❌ [CRITIC] GPT failed to review forecast for {market_id}")
                return None

            self.logger.info(
                f"🔍 [CRITIC] GPT review: {review.final_recommendation} "
                f"(agreement: {review.agreement_score:.1%}, approved: {review.approved})"
            )

            # Determine final decision
            total_cost = forecast.forecast_cost + review.review_cost

            if review.approved and forecast.action == "BUY":
                # Both agree to trade
                final_action = "BUY"

                # Use critic's adjusted confidence if provided
                if review.suggested_adjustments and 'adjusted_confidence' in review.suggested_adjustments:
                    final_confidence = review.suggested_adjustments['adjusted_confidence']
                else:
                    # Average the confidences weighted by agreement
                    final_confidence = (forecast.confidence + forecast.confidence * review.agreement_score) / 2

                dual_reasoning = (
                    f"[DUAL-AI APPROVED] Forecaster: {forecast.reasoning[:200]}... | "
                    f"Critic: {review.critique[:200]}..."
                )

                self.logger.info(
                    f"✅ [DUAL-AI] Trade APPROVED for {market_id}: "
                    f"{final_action} {forecast.side} @ {final_confidence:.1%} confidence"
                )

            else:
                # Critic rejected or forecaster said skip
                final_action = "SKIP"
                final_confidence = 0.0

                rejection_reason = review.rejection_reason or review.critique
                dual_reasoning = (
                    f"[DUAL-AI REJECTED] Critic: {rejection_reason[:300]}"
                )

                self.logger.info(
                    f"🚫 [DUAL-AI] Trade REJECTED for {market_id}: {rejection_reason[:100]}..."
                )

            return DualAIDecision(
                forecast=forecast,
                review=review,
                final_action=final_action,
                final_side=forecast.side,
                final_confidence=final_confidence,
                total_cost=total_cost,
                dual_ai_reasoning=dual_reasoning
            )

        except Exception as e:
            self.logger.error(f"❌ [DUAL-AI] Error in dual-AI decision: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    async def _get_grok_forecast(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str
    ) -> Optional[ForecastResult]:
        """
        Get a forecast from Grok (xAI) with evidence and reasoning.
        """
        try:
            # Calculate time to expiry
            import time
            expiration_ts = market_data.get('expiration_ts', time.time() + 86400)
            hours_to_expiry = max(0.1, (expiration_ts - time.time()) / 3600)
            days_to_expiry = hours_to_expiry / 24

            # Prepare the forecaster prompt
            prompt = GROK_FORECASTER_PROMPT.format(
                title=market_data.get('title', 'Unknown Market'),
                rules=market_data.get('rules', 'No specific rules provided'),
                yes_price=market_data.get('yes_price', 50),
                no_price=market_data.get('no_price', 50),
                volume=market_data.get('volume', 0),
                days_to_expiry=days_to_expiry,
                hours_to_expiry=hours_to_expiry,
                news_summary=news_summary or "No additional context available.",
                cash=portfolio_data.get('balance', portfolio_data.get('available_balance', 1000)),
                max_trade_value=portfolio_data.get('max_trade_value', 100)
            )

            # Call Grok via XAI client - handle both EnhancedAIClient wrapper and direct XAIClient
            messages = [{"role": "user", "content": prompt}]
            
            # Get the actual XAI client (may be wrapped in EnhancedAIClient)
            actual_xai_client = getattr(self.xai_client, 'xai_client', self.xai_client)
            
            response_text, cost = await actual_xai_client._make_completion_request(
                messages=messages,
                temperature=0.2,  # Low temperature for more consistent forecasts
                max_tokens=2000
            )

            if not response_text:
                return None

            # Parse the JSON response
            forecast_data = self._parse_json_response(response_text)

            if not forecast_data:
                self.logger.warning("Failed to parse forecaster JSON response")
                return None

            return ForecastResult(
                market_id=market_data.get('ticker', market_data.get('market_id', 'unknown')),
                predicted_probability=float(forecast_data.get('predicted_probability', 0.5)),
                confidence=float(forecast_data.get('confidence', 0.5)),
                side=str(forecast_data.get('side', 'YES')).upper(),
                action=str(forecast_data.get('action', 'SKIP')).upper(),
                evidence=forecast_data.get('evidence', []),
                key_factors=forecast_data.get('key_factors', []),
                risks=forecast_data.get('risks', []),
                reasoning=forecast_data.get('reasoning', 'No reasoning provided'),
                forecast_cost=cost
            )

        except Exception as e:
            self.logger.error(f"Error in Grok forecast: {e}")
            return None

    async def _get_gpt_review(
        self,
        market_data: Dict[str, Any],
        forecast: ForecastResult
    ) -> Optional[CriticReview]:
        """
        Get a critical review from GPT (OpenAI) of the forecast.
        """
        try:
            # Calculate time to expiry
            import time
            expiration_ts = market_data.get('expiration_ts', time.time() + 86400)
            hours_to_expiry = max(0.1, (expiration_ts - time.time()) / 3600)
            days_to_expiry = hours_to_expiry / 24

            # Format evidence for prompt
            evidence_str = "\n".join(f"- {e}" for e in forecast.evidence) if forecast.evidence else "No evidence provided"
            factors_str = "\n".join(f"- {f}" for f in forecast.key_factors) if forecast.key_factors else "No factors listed"
            risks_str = "\n".join(f"- {r}" for r in forecast.risks) if forecast.risks else "No risks identified"

            # Prepare the critic prompt
            prompt = GPT_CRITIC_PROMPT.format(
                title=market_data.get('title', 'Unknown Market'),
                yes_price=market_data.get('yes_price', 50),
                no_price=market_data.get('no_price', 50),
                volume=market_data.get('volume', 0),
                days_to_expiry=days_to_expiry,
                hours_to_expiry=hours_to_expiry,
                predicted_probability=forecast.predicted_probability,
                confidence=forecast.confidence,
                side=forecast.side,
                action=forecast.action,
                evidence=evidence_str,
                key_factors=factors_str,
                risks=risks_str,
                reasoning=forecast.reasoning
            )

            # Call GPT via OpenAI client - handle both EnhancedAIClient wrapper and direct OpenAIClient
            messages = [{"role": "user", "content": prompt}]
            
            # Get the actual OpenAI client (may be wrapped in EnhancedAIClient)
            actual_openai_client = getattr(self.openai_client, 'openai_client', self.openai_client)
            
            response_text, cost = await actual_openai_client._make_completion_request(
                messages=messages,
                temperature=0.3,  # Slightly higher for more critical thinking
                max_tokens=1500
            )

            if not response_text:
                return None

            # Parse the JSON response
            review_data = self._parse_json_response(response_text)

            if not review_data:
                self.logger.warning("Failed to parse critic JSON response")
                return None

            return CriticReview(
                approved=bool(review_data.get('approved', False)),
                agreement_score=float(review_data.get('agreement_score', 0.0)),
                critique=review_data.get('critique', 'No critique provided'),
                suggested_adjustments=review_data.get('suggested_adjustments'),
                final_recommendation=review_data.get('final_recommendation', 'REJECT'),
                rejection_reason=review_data.get('rejection_reason'),
                review_cost=cost
            )

        except Exception as e:
            self.logger.error(f"Error in GPT review: {e}")
            return None

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from an AI response, handling various formats with security safeguards.

        Security fixes implemented:
        1. No external json_repair library that could execute arbitrary code
        2. Limited JSON size to prevent memory exhaustion
        3. Strict validation of parsed structure
        4. Safe fallback without code execution
        """
        try:
            # Limit response text size to prevent memory exhaustion (1MB max)
            if len(response_text) > 1024 * 1024:
                self.logger.error("AI response too large for processing")
                return None

            # Try to extract JSON from code blocks first
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find raw JSON (limiting scope to prevent injection)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning("No JSON found in AI response")
                    return None

            # Basic validation before parsing
            if not json_str or len(json_str.strip()) < 2:
                self.logger.warning("Empty or invalid JSON string")
                return None

            # Additional security check - limit JSON string size
            if len(json_str) > 50000:  # 50KB limit
                self.logger.error("JSON content exceeds safe size limit")
                return None

            try:
                parsed_data = json.loads(json_str)

                # Validate the parsed JSON structure
                if not isinstance(parsed_data, dict):
                    self.logger.error("Parsed JSON is not a dictionary")
                    return None

                # Validate no potentially dangerous nested objects (functions, etc.)
                if self._contains_unsafe_types(parsed_data):
                    self.logger.error("Parsed JSON contains unsafe data types")
                    return None

                return parsed_data

            except json.JSONDecodeError as e:
                # Safe JSON repair without external libraries
                try:
                    # Simple character-level fixes for common issues
                    repaired = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    repaired = repaired.replace(',}', '}').replace(',]', ']')
                    repaired = re.sub(r',\s*,', ',', repaired)  # Remove double commas

                    parsed_data = json.loads(repaired)

                    # Validate again after repair
                    if not isinstance(parsed_data, dict) or self._contains_unsafe_types(parsed_data):
                        return None

                    self.logger.warning("JSON repair successful but may have altered data")
                    return parsed_data

                except (json.JSONDecodeError, ValueError) as repair_error:
                    self.logger.error(f"JSON repair failed: {repair_error}")
                    return None

        except Exception as e:
            self.logger.error(f"Error parsing JSON response: {e}")
            return None

    def _contains_unsafe_types(self, data: Any) -> bool:
        """
        Check if parsed data contains potentially unsafe types.
        """
        import types

        unsafe_types = (
            types.FunctionType,
            types.LambdaType,
            types.CodeType,
            types.GeneratorType,
            type,
            object.__class__,
        )

        if isinstance(data, unsafe_types):
            return True

        if isinstance(data, dict):
            for key, value in data.items():
                if self._contains_unsafe_types(key) or self._contains_unsafe_types(value):
                    return True
        elif isinstance(data, (list, tuple)):
            for item in data:
                if self._contains_unsafe_types(item):
                    return True

        return False

    async def close(self):
        """Close AI client connections."""
        if self._xai_initialized and self.xai_client:
            await self.xai_client.close()
        if self._openai_initialized and self.openai_client:
            await self.openai_client.close()
