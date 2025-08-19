# MLB Simulator

An end-to-end Python project to simulate MLB matchups.  
It fetches/cleans data, builds pitcher and batter profiles, and runs probability models (Poisson + Monte Carlo-style) to generate game-day insights such as strikeout distributions and batting prop probabilities.

---

## Features

### 🗄Data Layer & ETL
- Loads input data from `data/` (CSV-based) with **sample datasets included** for instant demo.
- ETL stubs provided for future integration with:
  - **Statcast** (batted-ball profiles, pitcher whiff rates)  
  - **FanGraphs** (advanced splits & projections)  
  - **MLB.com** (lineups, injuries, team depth charts)  
- Supports player lookup tables, lineup construction, and roster management.

###  Statistical Modeling
- **Pitcher Projections:** Poisson-based KO distributions from K/9 and IP/start.  
- **Batter Projections:** Per-PA event probabilities (1B, 2B, 3B, HR, BB).  
- **Monte Carlo Simulation (roadmap):** scalable engine to simulate thousands of games.  
- **Negative Binomial Extensions (roadmap):** improved variance handling for Ks and hits.  

###  Contextual Adjustments
- Lineup-spot based expected plate appearances (1–9).  
- Ready to incorporate **park factors, pace adjustments, and umpire tendencies**.  
- Extensible to account for **bullpen fatigue, travel effects, and injuries**.  

###  Outputs & Reporting
- **Starter KO probability tables** (e.g., P(K≥4), P(K≥6), 90% CIs).  
- **Batter prop tables** with P(Hits≥1), P(HR≥1), P(2B≥1), P(3B≥1).  
- Outputs printed to console in **clean tabular format**.  
- Future: export results to **CSV/Excel** for integration with dashboards or betting models.  

###  Extensible Design
- Modular architecture:  
  - `fetch.py` → handles ETL & data loading  
  - `precompute.py` → builds player profiles & probability models  
  - `gameday_sim.py` → orchestrates simulations & reporting  
- Easy to extend with:
  - **Regression models (scikit-learn, XGBoost)**  
  - **Bayesian updates** with live data  
  - **Full-game simulation with Markov chain or run expectancy matrices**  

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





