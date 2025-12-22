from src.config.settings import settings

def verify_sniper_mode():
    print("Verifying Sniper Mode Settings...")
    
    # Cost Controls
    assert settings.trading.daily_ai_budget <= 3.5, f"Budget too high: {settings.trading.daily_ai_budget}"
    assert settings.trading.max_ai_cost_per_decision <= 0.05, f"Cost per decision too high: {settings.trading.max_ai_cost_per_decision}"
    
    # Rate Limits
    assert settings.trading.scan_interval_seconds == 300, f"Scan interval mismatch: {settings.trading.scan_interval_seconds}"
    assert settings.trading.run_interval_minutes == 30, f"Run interval mismatch: {settings.trading.run_interval_minutes}"
    assert settings.trading.analysis_cooldown_hours == 12, f"Cooldown mismatch: {settings.trading.analysis_cooldown_hours}"
    
    # Capital Growth / Quality
    assert settings.trading.min_volume_for_analysis >= 1000.0, f"Analysis volume too low: {settings.trading.min_volume_for_analysis}"
    assert settings.trading.min_confidence_to_trade == 0.65, f"Confidence mismatch: {settings.trading.min_confidence_to_trade}"
    assert settings.trading.min_trade_edge >= 0.08, f"Edge mismatch: {settings.trading.min_trade_edge}"
    assert settings.trading.max_position_size_pct == 5.0, f"Position size mismatch: {settings.trading.max_position_size_pct}"
    assert settings.trading.max_positions == 4, f"Max positions mismatch: {settings.trading.max_positions}"
    
    # Market Making
    assert settings.trading.min_volume_for_market_making >= 2000.0, f"MM volume too low: {settings.trading.min_volume_for_market_making}"
    
    print("✅ All Sniper Mode settings verified successfully!")

if __name__ == "__main__":
    verify_sniper_mode()
