import numpy as np
import pandas as pd
from global_land_mask import globe
from functools import reduce
def hurdatclean(df,month1,month2,landmask=0,nearland=2):
    # Load HURDAT2-formatted hurricane data
    
    # --- Filtering: 6-hour advisories only & valid storm types ---
    
    # Boolean mask for every 6 hours (0, 600, 1200, 1800 UTC)
    six_hour_mask = df["Hours_in_UTC"].isin([0, 600, 1200, 1800])
    # Filter for tropical systems only (TS, HU, SS)
    storm_status_mask = df["Status_of_system"].isin(["TS", "HU", "SS"])

        
    # Combined final mask: only consider 6-hour advisories and tropical systems
    final_mask = six_hour_mask & storm_status_mask
    df = df[final_mask].reset_index(drop=True)
    # Fix Bonnie and Julia Effects on ACE in 2022, especially Bonnie
    df = df.drop([34685, 34686, 34691, 34692, 34693, 34694,
       34695, 34696, 34697, 34698, 34699, 34700, 34701, 34702, 34703, 34704,
       34705, 34706, 34707, 34708, 34709, 34710, 34711, 34712, 34713, 34714,
       34715, 34716, 34717, 34718, 34881, 34882])
    df = df.reset_index()
    if landmask==1:
        land_mask_array=[]
        for i in np.arange(-nearland,nearland+.1,0.1):
            for j in np.arange(-nearland,nearland+.1,0.1):
                new_data=pd.Series(globe.is_land(df['Latitude']+round(i,1),df['Longitude']+round(j,1)))
                land_mask_array.append(new_data)
        combined_mask_or = reduce(np.logical_or, land_mask_array)
        df = df[combined_mask_or].reset_index(drop=True)
    # --- Extract and clean data columns ---
    wind_vals = df["Maximum_sustained_wind_in_knots"].astype(int).to_numpy()
    month_vals = df["Month"].astype(int).to_numpy()
    year_vals = df["Year"].astype(int).to_numpy()
    storm_nums = df["ATCF_cyclone_number_for_that_year"].astype(int).to_numpy()
    storm_type = df["Status_of_system"].to_numpy()
    
    # --- Calculate unique storm counts per year ---
    # Track storm IDs by (year, storm number)
    storm_ids = list(zip(year_vals, storm_nums))
    storm_ids_2 = list(zip(year_vals, month_vals, storm_nums))
    storm_ids.append((2026,0))
    wind_vals=np.append(wind_vals, 0)
    unique_storms = []
    year_storm_count = []
    current_year = 1851
    counter = 0
    duration = 0
    storm_length=[]
    year_storm_length=[]
    for i in range(len(storm_ids)-1):
        y1, s1 = storm_ids[i]
        y2, s2 = storm_ids[i + 1]
        wind1=wind_vals[i]
        if s1 == s2:
            duration += 1
        if s1 != s2:
            counter += 1
            duration += 1
            storm_length.append(6*duration)
            duration = 0
            maxwind = 0
        if y1 != y2:
            year_storm_count.append(counter)
            year_storm_length.append(storm_length)
            counter = 0
            storm_length=[]
        if (y1 + 2 == y2):
            year_storm_count.append(0)
            year_storm_length.append([])
            counter = 0
            storm_length=[]            
    #year_storm_count.append(storm_nums[-1])  # Append last year
    
    # Manual data fixes (known edge cases in dataset)
    year_storm_count[63] += 1   # 1914
    year_storm_length[63].append(78)
    # yearmaxwinds[63].append(60)
    # year_storm_count[103] += 1  # 1954
    # year_storm_count[104] -= 1  # 1955
    # year_storm_length[103].append(year_storm_length[104][0])
    # del year_storm_length[104][0]
    # yearmaxwinds[103].append(yearmaxwinds[104][0])
    # del yearmaxwinds[104][0]

    # year_storm_count[154] += 1  # 2005
    # year_storm_count[155] -= 1  # 2006
    # year_storm_length[154].append(year_storm_length[155][0])
    # del year_storm_length[155][0]
    # yearmaxwinds[154].append(yearmaxwinds[155][0])
    # del yearmaxwinds[154][0]
    wind_vals = wind_vals[:-1]
    # --- ACE calculation utilities ---
    def ace_from_wind(wind):
        """Calculate ACE from one wind value in knots."""
        return (wind ** 2) / 10000
    
    def storm_duration(time):
        """Calculate ACE from one wind value in knots."""
        return 6*time
    
    def total_ace(month=None, year=None):
        """Return total ACE filtered by month and/or year."""
        mask = np.ones(len(wind_vals), dtype=bool)
        if month is not None:
            mask &= (month_vals == month)
        if year is not None:
            mask &= (year_vals == year)
        return ace_from_wind(wind_vals[mask]).sum()
    
    def total_duration(month=None, year=None):
        """Return total duration filtered by month and/or year."""
        mask = np.ones(len(wind_vals), dtype=bool)
        if month is not None:
            mask &= (month_vals == month)
        if year is not None:
            mask &= (year_vals == year)
        return storm_duration(len(storm_nums[mask]))#.sum()
    
    def total_maxwinds(month=None, year=None):
        """Return sum of maximum sustained winds in knots filtered by month and/or year."""
        mask = np.ones(len(wind_vals), dtype=bool)
        second_mask = np.ones(len(wind_vals), dtype=bool)
        third_mask = np.ones(len(wind_vals), dtype=bool)
        if month is not None:
            mask &= (month_vals == month)
            if month != 12:
                second_mask &= (month_vals == month+1)
            elif month == 12:
                second_mask &= (month_vals == month)
            if month != 1:
                third_mask &= (month_vals == month-1) 
            elif month == 1:
                third_mask &= (month_vals == month)
        if year is not None:
            mask &= (year_vals == year)
            if (month is not None) & (month != 12):
                second_mask &= (year_vals == year)
            elif (month is not None) & (month == 12):
                second_mask &= (year_vals == year+1)
            if (month is not None) & (month != 1):
                third_mask &= (year_vals == year)
            elif (month is not None) & (month == 1):
                third_mask &= (year_vals == year-1)
                
        jointnumwinds=np.array([storm_nums[mask],wind_vals[mask]]).T
        nextmonth=np.array([storm_nums[second_mask],wind_vals[second_mask]]).T
        pastmonth=np.array([storm_nums[third_mask],wind_vals[third_mask]]).T
        maximal_value=[]
        if np.size(storm_nums[mask]) == 0:
            return 0
        else:
            for i in range(storm_nums[mask][-1]):
                # print(i)
                subset=jointnumwinds[jointnumwinds[:,0] == i+1]
                if np.size(subset) == 0:
                    continue
                cleave=subset[:,1]
                maximumwinds=np.max(cleave)
                subset2=nextmonth[nextmonth[:,0] == i+1]
                if np.size(subset2) != 0:
                    cleave2=subset2[:,1]
                    maximumwinds2=np.max(cleave2)
                    if (maximumwinds2 > maximumwinds) & (month != 11):
                        continue
                subset3=pastmonth[pastmonth[:,0] == i+1]
                if np.size(subset3) != 0:
                    cleave3=subset3[:,1]
                    maximumwinds3=np.max(cleave3)
                    if (maximumwinds3 > maximumwinds) & (month != 6):
                        continue
                    else:
                        maximal_value.append(maximumwinds)
                else:
                    maximal_value.append(maximumwinds)
            return sum(maximal_value)
                
    # --- ACE and per-storm ACE by year ---
    yearz = np.arange(1851, 2026)
    yearly_ace = np.array([total_ace(year=yr) for yr in yearz])
    # To account for Alice and Zeta
    yearly_ace[103]= yearly_ace[103] + 6.12 # 1954
    yearly_ace[104]= yearly_ace[104] - 6.12 # 1955
    yearly_ace[154] = yearly_ace[154] + 4.795  # 2005
    yearly_ace[155] = yearly_ace[155] - 4.795  # 2006
    ace_per_storm = yearly_ace / np.array(year_storm_count)
    yearly_duration = np.array([total_duration(year=yr) for yr in yearz])
    #yearly_maxwind = [total_maxwinds(year=yr) for yr in yearz]

    # --- Monthly ACE average from 1951–2024 ---
    monthly_ace = np.zeros(12)
    for month in range(1, 13):
        monthly_ace[month - 1] = sum(total_ace(month=month, year=yr) for yr in range(1951, 2025))
    monthly_avg_ace = monthly_ace / 75  # 1951–2024 = 74 years
    
    # --- ACE by month for each year ---
    monthly_ace_by_year = np.array([[total_ace(month=m, year=y) for y in yearz] for m in range(1, 13)])
    monthly_duration_by_year = np.array([[total_duration(month=m, year=y) for y in yearz] for m in range(1, 13)])
    monthly_exlcusive_maximum_winds = np.array([[total_maxwinds(month=m, year=y) for y in yearz] for m in range(1, 13)])
    
    # --- Storm count by month per year (with and without splits) ---
    storm_counts_by_month = np.zeros((12, len(yearz)), dtype=int)
    storm_counts_nolap = np.zeros((12, len(yearz)), dtype=int)
    storm_counts_hurr = np.zeros((12, len(yearz)), dtype=int)
    storm_counts_mh = np.zeros((12, len(yearz)), dtype=int)
    
    storm_nums=np.append(storm_nums, 0)
    hurrcount=0
    mhcount=0
    for i in range(len(storm_nums) - 1):
        m, y = month_vals[i], year_vals[i]
        year_idx = y - 1851
        storm_id = storm_nums[i]
        stormtyper=storm_type[i]
    
        if hurrcount == 0 and stormtyper=="HU":
            storm_counts_hurr[m - 1, year_idx] += 1
            hurrcount=hurrcount+1
        if mhcount == 0 and stormtyper=="HU" and wind_vals[i] >= 100:
            storm_counts_mh[m - 1, year_idx] += 1
            mhcount=mhcount+1
        
        # Count storm-month (splitting storms across months)
        if storm_nums[i] != storm_nums[i + 1] or month_vals[i] != month_vals[i + 1]:
            storm_counts_by_month[m - 1, year_idx] += 1
    
        # Count storm (only first occurrence)
        if i == 0 or storm_nums[i] != storm_nums[i - 1]:
            storm_counts_nolap[m - 1, year_idx] += 1
            hurrcount=0
            mhcount=0
    
    
    # --- ACE per storm (monthly) ---
    ace_per_storm_month = np.nan_to_num(monthly_ace_by_year / storm_counts_by_month, nan=0.0)
    ace_per_storm_month_nolap = np.nan_to_num(monthly_ace_by_year / storm_counts_nolap, nan=0.0)
    seasonal_ace = monthly_ace_by_year[month1-1:month2].sum(axis=0)
    seasonal_num = storm_counts_nolap[month1-1:month2].sum(axis=0)
    seasonal_hurr = storm_counts_hurr[month1-1:month2].sum(axis=0)
    seasonal_mh = storm_counts_mh[month1-1:month2].sum(axis=0)
    seasonal_dur = monthly_duration_by_year[month1-1:month2].sum(axis=0)
    seasonal_maxwind = monthly_exlcusive_maximum_winds[month1-1:month2].sum(axis=0)
    if month1==6:
        seasonal_num[102] += 1
        seasonal_num[157] += 1
        seasonal_num[165] += 1

        
    seasonal_apsm = np.nan_to_num(seasonal_ace / seasonal_num, nan=0.0)
    seasonal_hurrrat = np.nan_to_num(seasonal_hurr / seasonal_num, nan=0.0)
    seasonal_mhrat = np.nan_to_num(seasonal_mh / seasonal_num, nan=0.0)
    seasonal_ap6h = np.nan_to_num(seasonal_ace / seasonal_dur, nan=0.0)
    seasonal_avdur = np.nan_to_num(seasonal_dur / seasonal_num, nan=0.0)
    seasonal_avmaxwind = np.nan_to_num(seasonal_maxwind / seasonal_num, nan=0.0)
    yearmask=df["Year"].isin(np.arange(1925, 2026))
    mod_df = df[yearmask].reset_index(drop=True)
    mod_df = mod_df.drop(["Year_",#'F34_kt_wind_radii_maximum_northeastern',
           # 'F34_kt_wind_radii_maximum_southeastern',
           # 'F34_kt_wind_radii_maximum_southwestern',
           # 'F34_kt_wind_radii_maximum_northwestern',
           'F50_kt_wind_radii_maximum_northeastern',
           'F50_kt_wind_radii_maximum_southeastern',
           'F50_kt_wind_radii_maximum_southwestern',
           'F50_kt_wind_radii_maximum_northwestern',
           'F64_kt_wind_radii_maximum_northeastern',
           'F64_kt_wind_radii_maximum_southeastern',
           'F64_kt_wind_radii_maximum_southwestern',
           'F64_kt_wind_radii_maximum_northwestern'],axis=1)
    mod_df["ACE"]=ace_from_wind(mod_df["Maximum_sustained_wind_in_knots"])
    return mod_df, yearly_ace, seasonal_num, seasonal_dur, seasonal_apsm, seasonal_ap6h, seasonal_avdur, monthly_ace_by_year, seasonal_avmaxwind, seasonal_hurrrat, seasonal_mhrat, seasonal_hurr, seasonal_mh, yearz