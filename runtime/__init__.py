"""
runtime/ — R2 live operation layer.

Modules:
  runtime_state     — central RuntimeState object
  scheduler         — periodic task and cooldown utilities
  update_cycle      — market data / indicator / strategy refresh
  commentary_cycle  — AI commentary generation with throttling
  user_input        — terminal command input handler
  runtime_loop      — main async orchestration loop
  run_r2            — startup entry point
"""