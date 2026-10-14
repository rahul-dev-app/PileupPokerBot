"""Write the 1000 benchmark decks (compare_agents.make_decks(1000, 0)) to /tmp/pp_decks.json for bench_search.js."""
import json

from compare_agents import make_decks

json.dump(make_decks(1000, 0).tolist(), open("/tmp/pp_decks.json", "w"))
