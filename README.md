# MLB Simulator

An end-to-end Python project to simulate MLB matchups.  
It fetches/cleans data, builds pitcher and batter profiles, and runs probability models (Poisson + Monte Carlo-style) to generate game-day insights such as strikeout distributions and batting prop probabilities.

---

## Features
- **Data layer (ETL-ready):** Loads CSVs from `data/` (sample files included). Stubs provided for Statcast/FanGraphs/MLB.com scrapers.
- **Modeling:** 
  - Pitcher strikeout projections (Poisson tails from K/9 + IP/start)  
  - Batter hit/HR probabilities from per-PA rates  
- **Outputs:** Starter KO probability tables, Batter prop tables (P(Hits≥1), P(HR≥1), etc.)
- **Extensible:** Easy to plug in park factors, bullpen logic, and advanced regression models.

```
nba_model/
├── fetch.py # Data layer: loads CSVs (or sample data), ETL stubs
├── precompute.py # Builds pitcher/batter profiles & probability models
├── data_fetch.py              # API and web scraping logic
├── gameday_sim.py # CLI driver – run simulations and print reports
├── requirements.txt # Python dependencies
├── LICENSE # MIT license
└── README.md # Project documentation
```

## Pipeline Flow
fetch.py ──► precompute.py ──► gameday_sim.py ──► Outputs
│ │ │
│ │ └── Prints starter KO tables & batter props
│ └── Builds pitcher & batter probability profiles
└── Loads CSV data (or sample), stubs for ETL
