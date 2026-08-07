import sqlite3
import pandas as pd

df = pd.read_csv('spacex_data_step2_enriched.csv')
conn = sqlite3.connect('spacex_capstone.db')
df.to_sql('SPACEXTBL', conn, if_exists='replace', index=False)

results = {}

def run_query(query, description):
    r = pd.read_sql_query(query, conn)
    results[description] = r
    print(f"\n--- {description} ---")
    print(r.to_string(index=False))
    return r

run_query("SELECT DISTINCT LaunchSite FROM SPACEXTBL;", "Unique launch sites")
run_query("SELECT * FROM SPACEXTBL WHERE LaunchSite LIKE 'CCA%' LIMIT 5;", "First 5 launches from a CCA-prefixed site")
run_query("SELECT SUM(PayloadMass) AS Total_Payload_Mass FROM SPACEXTBL;", "Total payload mass across all launches")
run_query("SELECT AVG(PayloadMass) AS Average_Payload_Mass FROM SPACEXTBL WHERE BoosterVersion = 'Falcon 9';", "Average payload mass for Falcon 9")
run_query("SELECT MIN(Date) AS First_Successful_Landing_Date FROM SPACEXTBL WHERE Class = 1 AND LandingPad IS NOT NULL;", "Date of first successful ground pad landing")
run_query("""SELECT Serial FROM SPACEXTBL WHERE Class = 1 AND LandingPad IS NOT NULL
             AND PayloadMass > 4000 AND PayloadMass < 6000;""", "Boosters with successful drone ship landing and payload 4000-6000kg")
run_query("SELECT Outcome, COUNT(*) AS Count FROM SPACEXTBL GROUP BY Outcome;", "Mission outcome counts")
run_query("SELECT LaunchSite, Outcome, COUNT(*) as Count FROM SPACEXTBL GROUP BY LaunchSite, Outcome;", "Mission outcome counts per launch site")
run_query("SELECT BoosterVersion, MAX(PayloadMass) AS Max_Payload FROM SPACEXTBL WHERE PayloadMass = (SELECT MAX(PayloadMass) FROM SPACEXTBL);", "Booster with maximum payload mass")

conn.close()

# Save summary text
with open('sql_results_summary.txt', 'w') as f:
    for k, v in results.items():
        f.write(f"=== {k} ===\n")
        f.write(v.to_string(index=False))
        f.write("\n\n")

print("\nSQL results saved to sql_results_summary.txt")
