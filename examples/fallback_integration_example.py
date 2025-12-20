"""
Integration example showing how to use the Enhanced Fallback and Redundancy Systems.

This example demonstrates:
- Setting up multi-provider redundancy
- Using the enhanced client with fallbacks
- Handling emergency modes
- Monitoring system health
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Note: In a real application, you would import the actual classes
# from src.intelligence.enhanced_client import EnhancedAIClient
# from src.intelligence.fallback_manager import ProviderConfig
# from src.intelligence.provider_manager import ProviderType


async def demonstrate_multi_provider_redundancy():
    """Demonstrate multi-provider redundancy setup."""
    print("🔄 Demonstrating Multi-Provider Redundancy")
    print("=" * 50)

    # Example provider configurations
    providers = {
        "xai": {
            "name": "xai",
            "endpoint": "https://api.x.ai/v1",
            "api_key": "your-xai-api-key",
            "models": ["grok-4", "grok-3"],
            "priority": 1,
            "timeout": 30.0,
            "max_retries": 3,
            "cost_per_token": 0.000015
        },
        "openai": {
            "name": "openai",
            "endpoint": "https://api.openai.com/v1",
            "api_key": "your-openai-api-key",
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "priority": 2,
            "timeout": 25.0,
            "max_retries": 3,
            "cost_per_token": 0.00001
        },
        "anthropic": {
            "name": "anthropic",
            "endpoint": "https://api.anthropic.com/v1",
            "api_key": "your-anthropic-api-key",
            "models": ["claude-3-opus", "claude-3-sonnet"],
            "priority": 3,
            "timeout": 35.0,
            "max_retries": 2,
            "cost_per_token": 0.000025
        },
        "local": {
            "name": "local",
            "endpoint": "http://localhost:11434/v1",
            "api_key": "local-key",
            "models": ["llama-2", "mistral"],
            "priority": 4,
            "timeout": 60.0,
            "max_retries": 1,
            "cost_per_token": 0.000001
        }
    }

    print("📋 Provider Configuration:")
    for name, config in providers.items():
        print(f"  • {name.capitalize()}:")
        print(f"    - Priority: {config['priority']}")
        print(f"    - Models: {', '.join(config['models'])}")
        print(f"    - Cost per token: ${config['cost_per_token']:.6f}")
        print(f"    - Timeout: {config['timeout']}s")

    # In real usage, this would create the enhanced client
    print(f"\n✅ Configured {len(providers)} providers for redundancy")
    return providers


async def demonstrate_trading_decisions():
    """Demonstrate getting trading decisions with fallbacks."""
    print("\n🤖 Demonstrating Enhanced Trading Decision Flow")
    print("=" * 50)

    # Example market and portfolio data
    market_data = {
        "title": "Will the S&P 500 close above 5000 by end of year?",
        "yes_price": 65,
        "no_price": 35,
        "volume": 50000,
        "days_to_expiry": 45,
        "category": "indices"
    }

    portfolio_data = {
        "balance": 10000,
        "max_trade_value": 500,
        "positions": []
    }

    news_summary = """
    Recent market analysis shows positive sentiment towards tech stocks,
    with major indices reaching new highs. Economic indicators suggest
    continued growth trajectory.
    """

    print("📊 Sample Market Data:")
    print(f"  • Market: {market_data['title']}")
    print(f"  • YES Price: {market_data['yes_price']}¢ | NO Price: {market_data['no_price']}¢")
    print(f"  • Volume: ${market_data['volume']:,}")
    print(f"  • Days to expiry: {market_data['days_to_expiry']}")

    print(f"\n💰 Portfolio State:")
    print(f"  • Available balance: ${portfolio_data['balance']:,}")
    print(f"  • Max trade value: ${portfolio_data['max_trade_value']}")

    # Decision flow with fallbacks
    print(f"\n🔄 Enhanced Decision Flow:")

    print("1️⃣  Trying primary provider (xAI)...")
    # In real usage: decision = await enhanced_client.get_trading_decision(market_data, portfolio_data, news_summary)
    print("   ✅ xAI responded with decision")
    print("   📊 Action: BUY | Side: YES | Confidence: 78%")
    print("   💰 Cost: $0.023 | Response time: 1.2s")

    # Simulate provider failure
    print("\n2️⃣  Simulating xAI provider failure...")
    print("   ❌ xAI API timeout (30s)")

    print("\n3️⃣  Initiating failover to OpenAI...")
    print("   🔍 Checking OpenAI health: Healthy")
    print("   ✅ OpenAI responded with decision")
    print("   📊 Action: BUY | Side: YES | Confidence: 72%")
    print("   💰 Cost: $0.015 | Response time: 2.1s")

    # Simulate complete provider outage
    print("\n4️⃣  Simulating complete provider outage...")
    print("   ❌ All providers unavailable")

    print("\n5️⃣  Activating Emergency Mode...")
    print("   🚨 Emergency mode activated for 60 minutes")
    print("   🛡️  Using conservative trading strategy")
    print("   📊 Action: SKIP | Side: YES | Confidence: 30%")
    print("   💡 Reasoning: Emergency conservative approach - no trading during outage")


async def demonstrate_health_monitoring():
    """Demonstrate health monitoring and system status."""
    print("\n🏥 Demonstrating Health Monitoring System")
    print("=" * 50)

    # Simulate health check results
    health_status = {
        "xai": {
            "status": "healthy",
            "response_time": 120.5,
            "last_check": "2025-12-19T23:45:00Z",
            "success_rate": 98.5
        },
        "openai": {
            "status": "degraded",
            "response_time": 850.2,
            "last_check": "2025-12-19T23:45:00Z",
            "success_rate": 95.2
        },
        "anthropic": {
            "status": "healthy",
            "response_time": 200.1,
            "last_check": "2025-12-19T23:45:00Z",
            "success_rate": 99.1
        },
        "local": {
            "status": "unhealthy",
            "response_time": 0.0,
            "last_check": "2025-12-19T23:45:00Z",
            "success_rate": 45.3
        }
    }

    print("📊 Current Provider Health Status:")
    for provider, status in health_status.items():
        status_icon = "✅" if status["status"] == "healthy" else "⚠️" if status["status"] == "degraded" else "❌"

        print(f"  {status_icon} {provider.upper()}:")
        print(f"    - Status: {status['status']}")
        print(f"    - Response time: {status['response_time']:.1f}ms")
        print(f"    - Success rate: {status['success_rate']:.1f}%")
        print(f"    - Last check: {status['last_check']}")

    # System status summary
    healthy_count = sum(1 for s in health_status.values() if s["status"] == "healthy")
    total_count = len(health_status)
    degradation_level = (total_count - healthy_count) / total_count

    print(f"\n📈 System Status:")
    print(f"  • Mode: {'Normal' if degradation_level == 0 else 'Degraded' if degradation_level < 1 else 'Emergency'}")
    print(f"  • Healthy providers: {healthy_count}/{total_count}")
    print(f"  • Degradation level: {degradation_level:.1%}")
    print(f"  • Available for trading: {healthy_count > 0}")


async def demonstrate_emergency_modes():
    """Demonstrate different emergency mode levels."""
    print("\n🚨 Demonstrating Emergency Trading Modes")
    print("=" * 50)

    emergency_modes = [
        {
            "name": "Conservative Mode",
            "description": "Reduced position sizes, higher confidence thresholds",
            "rules": [
                "Only trade if confidence > 80%",
                "Maximum position: 2% of portfolio",
                "Skip volatile markets",
                "Use cached decisions when possible"
            ]
        },
        {
            "name": "Minimal Mode",
            "description": "Very limited trading, extreme risk aversion",
            "rules": [
                "Only trade if confidence > 95%",
                "Maximum position: 1% of portfolio",
                "Skip all high-volume markets",
                "Only trade cached decisions"
            ]
        },
        {
            "name": "Suspended Mode",
            "description": "No active trading, monitoring only",
            "rules": [
                "No new trades initiated",
                "Monitor system health continuously",
                "Alert on provider recovery",
                "Maintain emergency decision cache"
            ]
        }
    ]

    for mode in emergency_modes:
        print(f"📋 {mode['name']}:")
        print(f"  💡 {mode['description']}")
        print("  🔧 Rules:")
        for rule in mode['rules']:
            print(f"    • {rule}")
        print()


async def demonstrate_cost_optimization():
    """Demonstrate cost optimization and budget management."""
    print("\n💰 Demonstrating Cost Optimization")
    print("=" * 50)

    # Simulate cost tracking
    cost_data = {
        "xai": {
            "requests": 145,
            "tokens": 72450,
            "cost": 1.09,
            "avg_response_time": 1.2
        },
        "openai": {
            "requests": 98,
            "tokens": 42100,
            "cost": 0.42,
            "avg_response_time": 2.1
        },
        "anthropic": {
            "requests": 23,
            "tokens": 18750,
            "cost": 0.47,
            "avg_response_time": 1.8
        },
        "local": {
            "requests": 45,
            "tokens": 31200,
            "cost": 0.03,  # Essentially free
            "avg_response_time": 8.5
        }
    }

    print("💸 Cost Breakdown by Provider:")
    total_cost = 0
    total_requests = 0

    for provider, data in cost_data.items():
        cost_per_request = data["cost"] / data["requests"] if data["requests"] > 0 else 0
        total_cost += data["cost"]
        total_requests += data["requests"]

        print(f"  💡 {provider.upper()}:")
        print(f"    • Requests: {data['requests']}")
        print(f"    • Tokens: {data['tokens']:,}")
        print(f"    • Cost: ${data['cost']:.2f}")
        print(f"    • Cost per request: ${cost_per_request:.4f}")
        print(f"    • Avg response time: {data['avg_response_time']}s")

    print(f"\n📈 Cost Optimization Summary:")
    print(f"  • Total requests: {total_requests:,}")
    print(f"  • Total cost: ${total_cost:.2f}")
    print(f"  • Average cost per request: ${total_cost/total_requests:.4f}")
    print(f"  • Cost savings from local model: ${(cost_data['xai']['cost'] - cost_data['local']['cost']):.2f}")

    # Budget management example
    daily_budget = 50.0
    remaining_budget = daily_budget - total_cost
    budget_usage = (total_cost / daily_budget) * 100

    print(f"\n💳 Budget Management:")
    print(f"  • Daily budget: ${daily_budget:.2f}")
    print(f"  • Used so far: ${total_cost:.2f} ({budget_usage:.1f}%)")
    print(f"  • Remaining: ${remaining_budget:.2f}")
    print(f"  • Status: {'✅ Within budget' if remaining_budget > 0 else '⚠️ Budget exceeded'}")


async def main():
    """Main demonstration function."""
    print("🚀 Enhanced Fallback and Redundancy Systems - Integration Example")
    print("=" * 70)
    print("This example demonstrates the comprehensive fallback and redundancy")
    print("capabilities implemented for the Kalshi AI Trading Bot.")
    print()

    await demonstrate_multi_provider_redundancy()
    await demonstrate_trading_decisions()
    await demonstrate_health_monitoring()
    await demonstrate_emergency_modes()
    await demonstrate_cost_optimization()

    print("\n" + "=" * 70)
    print("🎉 Integration Example Complete!")
    print("\n📚 Key Benefits of the Enhanced Fallback System:")
    print("• ✅ Zero single points of failure with multi-provider redundancy")
    print("• ✅ Graceful degradation maintains system during partial outages")
    print("• ✅ Emergency modes ensure continued operation during major issues")
    print("• ✅ Comprehensive health monitoring and proactive failover")
    print("• ✅ Cost optimization with intelligent provider selection")
    print("• ✅ Seamless integration with existing trading infrastructure")
    print("• ✅ Conservative decision making in emergency scenarios")
    print("• ✅ Automatic recovery and system self-healing")
    print("• ✅ Comprehensive metrics and performance tracking")

    print("\n🔧 Ready for Production Integration:")
    print("1. Configure API keys for desired providers")
    print("2. Add EnhancedAIClient to your trading job logic")
    print("3. Configure health monitoring intervals")
    print("4. Set up cost optimization thresholds")
    print("5. Enable emergency mode notifications")
    print("6. Monitor system health and performance")

    return 0


if __name__ == "__main__":
    asyncio.run(main())