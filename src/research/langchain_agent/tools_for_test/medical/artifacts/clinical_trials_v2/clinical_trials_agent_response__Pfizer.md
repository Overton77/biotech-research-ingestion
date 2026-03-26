Summary of actions and results:

1) How to search for lead sponsor "Pfizer"
- Quick API parameter: use query.lead=Pfizer (preferred for lead-sponsor/company searches).
- Sponsor/collaborator parameter also works: query.spons=Pfizer.
- Advanced AREA[...] example (same effect): query.term=AREA[LeadSponsorName]Pfizer
  - You can combine AREA clauses, e.g. AREA[LeadSponsorName]Pfizer AND AREA[StudyType]INTERVENTIONAL
- Notes: LeadSponsorName may include legacy company names (e.g., "Wyeth is now a wholly owned subsidiary of Pfizer"), so consider searching broader sponsor text with query.spons or query.term if you want subsidiaries included.

2) Found studies (up to 3) with lead sponsor Pfizer
- NCT00396799 — Lead sponsor: "Wyeth is now a wholly owned subsidiary of Pfizer" — Status: COMPLETED
- NCT00580632 — Lead sponsor: "Wyeth is now a wholly owned subsidiary of Pfizer" — Status: COMPLETED
- NCT00926263 — Lead sponsor: "Pfizer" — Status: TERMINATED

(Raw and formatted search artifacts saved to:
C:\Users\Pinda\Proyectos\Biotech\biotech-research-ingestion\src\research\langchain_agent\tools_for_test\medical\artifacts\clinical_trials_v2\clinical_trials_search_raw__Pfizer.json
C:\Users\Pinda\Proyectos\Biotech\biotech-research-ingestion\src\research\langchain_agent\tools_for_test\medical\artifacts\clinical_trials_v2\clinical_trials_search_formatted__Pfizer.md)

3) Downloaded one full NCT record
- Downloaded NCT00926263 full study JSON.
- Exact download path:
C:\Users\Pinda\Proyectos\Biotech\biotech-research-ingestion\src\research\langchain_agent\tools_for_test\medical\artifacts\clinical_trials_v2\NCT00926263.json

If you want, I can open and summarize the downloaded study JSON fields (sponsor, interventions, status, enrollment, locations, key dates).