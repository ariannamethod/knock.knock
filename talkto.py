#!/usr/bin/env python3
# talkto.py — Async Router for HAZE + CLOUD
#
# Unified async interface:
# - Default: HAZE only (fast, autonomous)
# - /cloud: Toggle CLOUD mode (pre-semantic sonar)
# - /stats: Show bridge statistics
#
# Design principle: HAZE IS ALWAYS AUTONOMOUS
# - CLOUD is optional enhancement
# - Silent fallback if CLOUD fails
# - No errors leak to user
#
# "Two minds that can resonate together,
#  but never depend on each other."

import sys
import asyncio
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "haze"))

from bridge import AsyncBridge, HAS_CLOUD, HAS_HAZE


class AsyncHazeCloudRouter:
    """
    Async router that orchestrates HAZE and CLOUD.
    
    HAZE is always primary (autonomous voice generation).
    CLOUD is optional (pre-semantic emotion detection).
    
    Commands:
        /cloud   - toggle CLOUD mode
        /stats   - show bridge statistics  
        /help    - show help
        /quit    - exit
    
    Silent fallback: if CLOUD fails, HAZE continues without error messages.
    """
    
    def __init__(self):
        self.bridge: AsyncBridge = None
        self.cloud_enabled = False
        self._initialized = False
    
    async def initialize(self):
        """Initialize the bridge."""
        print("=" * 60)
        print("  ██╗  ██╗ █████╗ ███████╗███████╗")
        print("  ██║  ██║██╔══██╗╚══███╔╝██╔════╝")
        print("  ███████║███████║  ███╔╝ █████╗  ")
        print("  ██╔══██║██╔══██║ ███╔╝  ██╔══╝  ")
        print("  ██║  ██║██║  ██║███████╗███████╗")
        print("  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝")
        print("=" * 60)
        print()
        print("  HAZE + CLOUD Router (Async)")
        print()
        print(f"  HAZE available: {'✓' if HAS_HAZE else '✗'}")
        print(f"  CLOUD available: {'✓' if HAS_CLOUD else '✗'}")
        print()
        
        # Create bridge
        corpus_path = Path("haze/text.txt")
        if not corpus_path.exists():
            corpus_path = Path(__file__).parent / "haze" / "text.txt"
        
        self.bridge = await AsyncBridge.create(
            corpus_path=str(corpus_path),
            enable_cloud=True,
            silent_fallback=True,
            cloud_timeout=0.5,
        )
        
        self._initialized = True
        
        print(f"  Bridge initialized:")
        print(f"    HAZE: {'✓ ready' if self.bridge.haze else '✗ not available'}")
        print(f"    CLOUD: {'✓ ready' if self.bridge.cloud else '✗ not available (silent fallback)'}")
        print()
        print("  Commands:")
        print("    /cloud  - toggle CLOUD emotion detection")
        print("    /stats  - show statistics")
        print("    /help   - show all commands")
        print("    /quit   - exit")
        print()
        print("  Mode: HAZE only (type /cloud to enable emotion detection)")
        print("=" * 60)
        print()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.bridge:
            await self.bridge.__aexit__(None, None, None)
    
    def toggle_cloud(self):
        """Toggle CLOUD mode."""
        self.cloud_enabled = not self.cloud_enabled
        
        if self.cloud_enabled:
            if self.bridge.cloud:
                print("✓ CLOUD enabled (pre-semantic emotion detection)")
            else:
                print("⚠ CLOUD requested but not available (silent fallback active)")
        else:
            print("✗ CLOUD disabled (HAZE only mode)")
    
    def show_stats(self):
        """Show bridge statistics."""
        stats = self.bridge.stats()
        
        print("=" * 60)
        print("  Bridge Statistics")
        print("=" * 60)
        print()
        print(f"  HAZE: {'✓ active' if stats['haze_available'] else '✗ not available'}")
        print(f"  CLOUD: {'✓ active' if stats['cloud_available'] else '✗ not available'}")
        print()
        
        if stats['cloud_successes'] + stats['cloud_failures'] + stats['cloud_timeouts'] > 0:
            print("  CLOUD stats:")
            print(f"    Successes: {stats['cloud_successes']}")
            print(f"    Failures: {stats['cloud_failures']}")
            print(f"    Timeouts: {stats['cloud_timeouts']}")
            print(f"    Success rate: {stats['cloud_success_rate']:.1%}")
        else:
            print("  CLOUD stats: no requests yet")
        
        print()
        print("=" * 60)
    
    def show_help(self):
        """Show help."""
        print()
        print("Commands:")
        print("  /cloud  - toggle CLOUD emotion detection")
        print("  /stats  - show bridge statistics")
        print("  /help   - show this help")
        print("  /quit   - exit")
        print()
        print("Just type anything to talk to HAZE.")
        if self.cloud_enabled:
            print("CLOUD will detect emotions before HAZE responds.")
        print()
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        response = await self.bridge.respond(
            user_input,
            use_cloud=self.cloud_enabled,
        )
        
        # Show CLOUD info if enabled and available
        if self.cloud_enabled and response.cloud_hint:
            hint = response.cloud_hint
            print(f"  [cloud] {hint.primary} + {hint.secondary}", end="")
            if hint.anomaly.has_anomaly:
                print(f" | {hint.anomaly.anomaly_type}", end="")
            print()
        
        return response.text
    
    async def interactive_loop(self):
        """Main interactive loop."""
        if not self._initialized:
            await self.initialize()
        
        while True:
            try:
                # Get input
                try:
                    user_input = input("[you] ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    
                    if cmd == "/cloud":
                        self.toggle_cloud()
                        continue
                    
                    if cmd in ["/stats", "/stat"]:
                        self.show_stats()
                        continue
                    
                    if cmd in ["/help", "/h", "/?"]:
                        self.show_help()
                        continue
                    
                    if cmd in ["/quit", "/q", "/exit"]:
                        print("Goodbye! The haze settles...")
                        break
                    
                    print(f"Unknown command: {user_input}")
                    print("Type /help for available commands")
                    continue
                
                # Process input
                response = await self.process_input(user_input)
                print(f"[haze] {response}")
                print()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! The haze settles...")
                break
            
            except Exception as e:
                print(f"[error] {e}")
                continue
        
        await self.cleanup()


async def main():
    """Entry point."""
    router = AsyncHazeCloudRouter()
    await router.initialize()
    await router.interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())
