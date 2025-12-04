Scripts within the "Python Scripts" folder should be run in the following order:  
"NHL_API_Scraper.py"  
"CapWages_Scraper.py"  
"Map_NST_API_GameIDs.py"  
"Clean_Data.py"  
"RAPM_Calculator.py"  
"GAR_WAR_SPAR_Calculator.py"  
"WAR_value_conversion.py"  
"age_curve.py"  
"contract_prediction.py"  
"dashboard.py"  
  
You do not need to specify any command line arguments when running, and the pathing for files should hold as long as the file structure remains as is.  
Here is a rough ASCII layout of the project files in my working folder:  
  
Root/  
├─ Chrome/  
│  ├─ chrome-win64/  
│  ├─ chromedriver-win64/  
├─ Data/  
│  ├─ Clean Data/  
│  │  ├─ (Seasonal Subfolders)/  
│  │  ├─ <Cleaned League and Age Curve Files>  
│  ├─ Raw Data/  
│  │  ├─ HockeyReference/  
│  │  ├─ MoneyPuck/  
│  │  │  ├─ Team Level/  
│  │  ├─ NaturalStatTrick/  
├─ Python Scripts/  
│  ├─ <python script files>  
│  ├─ visualizations + helpers/  
│  │  ├─ <visualization scripts>  
├─ README.md  
├─ .gitattributes  

