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
```
fetch.py ──► precompute.py ──► gameday_sim.py ──► Outputs
│ │ │
│ │ └── Prints starter KO tables & batter props
│ └── Builds pitcher & batter probability profiles
└── Loads CSV data (or sample), stubs for ETL
```
## Example Outputs 
[FDodds.csv](https://github.com/user-attachments/files/21846368/FDodds.csv)
Utilize Sportsbook odds to a csv file to create baseline modeling 
<img width="167" height="744" alt="image" src="https://github.com/user-attachments/assets/46a8e5fc-4aad-4e16-b368-c586219d6806" />

Select Today's Matchup
<img width="1314" height="291" alt="image" src="https://github.com/user-attachments/assets/dde6ee08-855e-4e79-b432-b9167e9f2b7c" />

Simulate and creates 
Prints starter KO tables & batter props
Builds pitcher & batter probability profiles
Loads CSV data (or sample), stubs for ETL 
<img width="1483" height="447" alt="image" src="https://github.com/user-attachments/assets/192806e6-298c-4797-8fe3-d36b182ecece" />





