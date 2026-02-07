17-01-2026
- Meet team members onsite in Düsseldorf
- Discuss Project Ideas
20-01-2026
- Find dataset in Kaggle and others 
- Collected 9 datasets that possibly be used (all from kaggle)
21-01-2026 
- Online meeting with team 1
- Discuss about Project Structure 
- Which questions should be answered by the end of the Project? 
22-01-2026
- Meeting with Gizem (onl) and together prepare for Slide Project Pitch presentation 
24-01-2026
- Onsite (Düss) Pitch presentation 
- Create Github Reporatory for the Team's project
- Each member choose a Github's link to crape data
26-01-2026
- Clone JobMiner (GitHub) and check out the code
- Code crack, try to fix 
- Code run but only get data from demo version 
28-01-2026
- Try manual code for each Platform (LinkedIn) to fetch real data 
- Get the data for some jobs (data analyst, data scientist) in some regions (NRW, Dortmund, Düsseldorf) but still get error in NRW region (job from other regions show up)
- Job fetch limits each time 10 jobs (rows) 
29-01-2026
- Team meering (onl) to share scrape data results
- Finalize list of jobs and cities 
- Introduced about Log file
02-02-2026
- Create new branch in Github that ready to push
- Rewrite code to crape data in a correct list of jobs and cities
- After some tries and get block IP address
04-02-2026
- Use VPN to change IP and continues with the code
- Read Iebo code and try to create Log files like that
- Get some crash
05-02-2026
- Meeting onl with Gizem for the slide presentation this Saturday (7/2)
- Create slide together
- Try to debug the code
- Give more information to fetch data
- Limit target_time_range_days (120 days = 3 months)
- 1st estimated time scrape too long (20 hours), try to reduce the time
06-02-2026 
- Add auto checkpoint to save data in case code crash 
- Use Multi-threadng to reduce time scrapping 
- Beeing blocked IP again by LinkedIn after get checkpoint 1200 results 
- Try to scrape again by continue after checkpoints 1200 but get only 202 unique jobs (too small for a dataset)
- Try to fix the code again and rerun, stop Multi_threading to avoid beeing blocked by LinkedIn again 
- Estimated crapping time = 12.2 hours. Take a try for safe results. 
- Bug: missing Year filter, add code for year filter
- Clean result (create clean_old_jobs.py) and save result in 3 files (Metadata..., Raw_Jobs..._clean.json and Raw_Jobs..._clean.csv)
- Update the document journeys
