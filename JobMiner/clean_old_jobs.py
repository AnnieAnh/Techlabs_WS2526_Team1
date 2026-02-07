"""
Clean old jobs (>120 days) from LinkedIn scraper output.
Removes jobs with 'year' in posted_date field.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import re

def parse_posted_date_to_days(posted_date_str):
    """
    Parse LinkedIn posted date string to number of days ago.
    Includes handling for years (which was missing in original scraper).
    """
    if not posted_date_str or pd.isna(posted_date_str):
        return None
    
    posted_date_str = str(posted_date_str).lower()
    
    # Hours
    if 'hour' in posted_date_str or 'hr' in posted_date_str:
        return 0
    
    # Days
    match = re.search(r'(\d+)\s*day', posted_date_str)
    if match:
        return int(match.group(1))
    
    # Weeks
    match = re.search(r'(\d+)\s*week', posted_date_str)
    if match:
        return int(match.group(1)) * 7
    
    # Months
    match = re.search(r'(\d+)\s*month', posted_date_str)
    if match:
        return int(match.group(1)) * 30
    
    # Years (this was missing!)
    match = re.search(r'(\d+)\s*year', posted_date_str)
    if match:
        return int(match.group(1)) * 365
    
    return None

def clean_csv_file(input_csv, output_csv, target_days=120):
    """Clean CSV file by removing jobs older than target_days."""
    
    print(f"\n📂 Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    print(f"📊 Initial jobs: {initial_count}")
    
    # Add days_ago column if not exists
    if 'days_ago' not in df.columns or df['days_ago'].isna().all():
        print("\n🔄 Calculating days_ago for all jobs...")
        df['days_ago'] = df['posted_date'].apply(parse_posted_date_to_days)
    
    # Filter jobs within target_days
    print(f"\n🔍 Filtering jobs within {target_days} days...")
    
    # Remove jobs where days_ago > target_days
    df_filtered = df[
        (df['days_ago'].isna()) |  # Keep jobs where we can't parse date (to be safe)
        (df['days_ago'] <= target_days)
    ].copy()
    
    # Additional check: remove any job with 'year' in posted_date
    df_filtered = df_filtered[~df_filtered['posted_date'].str.contains(r'year', case=False, na=False)]
    
    final_count = len(df_filtered)
    removed_count = initial_count - final_count
    
    print(f"\n✅ Jobs within {target_days} days: {final_count}")
    print(f"❌ Jobs removed (too old): {removed_count}")
    print(f"📈 Retention rate: {final_count/initial_count*100:.1f}%")
    
    # Save cleaned CSV
    print(f"\n💾 Saving cleaned CSV: {output_csv}")
    df_filtered.to_csv(output_csv, index=False)
    
    return df_filtered, removed_count

def clean_json_file(input_json, output_json, valid_urls):
    """Clean JSON file by keeping only jobs that are in valid_urls."""
    
    print(f"\n📂 Reading: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    
    initial_count = len(jobs)
    print(f"📊 Initial jobs: {initial_count}")
    
    # Filter jobs
    jobs_filtered = [job for job in jobs if job.get('job_url') in valid_urls]
    
    final_count = len(jobs_filtered)
    removed_count = initial_count - final_count
    
    print(f"\n✅ Jobs kept: {final_count}")
    print(f"❌ Jobs removed: {removed_count}")
    
    # Save cleaned JSON
    print(f"\n💾 Saving cleaned JSON: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(jobs_filtered, f, indent=2, ensure_ascii=False)
    
    return jobs_filtered

def update_metadata(metadata_file, old_count, new_count, removed_count):
    """Update metadata file with new statistics."""
    
    print(f"\n📋 Updating metadata: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Update counts
    metadata['jobs_before_cleaning'] = old_count
    metadata['jobs_after_cleaning'] = new_count
    metadata['jobs_removed_old'] = removed_count
    metadata['cleaned_at'] = datetime.now().isoformat()
    metadata['cleaning_reason'] = 'Removed jobs older than 120 days (year-old jobs that were not filtered during scraping)'
    
    # Save updated metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("✅ Metadata updated")

def main():
    # File paths
    base_dir = Path(r"e:\DS\Project-Group-1\Techlabs_WS2526_Team1\JobMiner")
    timestamp = "2026-02-07_01-03-51"
    
    input_csv = base_dir / f"Raw_Jobs_LINKEDIN_{timestamp}.csv"
    input_json = base_dir / f"Raw_Jobs_LINKEDIN_{timestamp}.json"
    metadata_file = base_dir / f"Metadata_{timestamp}.json"
    
    # Output files (replace originals)
    output_csv = base_dir / f"Raw_Jobs_LINKEDIN_{timestamp}_cleaned.csv"
    output_json = base_dir / f"Raw_Jobs_LINKEDIN_{timestamp}_cleaned.json"
    
    print("="*80)
    print("🧹 CLEANING OLD JOBS FROM LINKEDIN DATA")
    print("="*80)
    print(f"📅 Target: Last 120 days only")
    print(f"🎯 Action: Remove jobs with 'year' in posted_date")
    print("="*80)
    
    # Clean CSV
    df_cleaned, removed_count = clean_csv_file(input_csv, output_csv, target_days=120)
    
    # Get valid job URLs from cleaned data
    valid_urls = set(df_cleaned['job_url'].values)
    
    # Clean JSON
    jobs_cleaned = clean_json_file(input_json, output_json, valid_urls)
    
    # Update metadata
    if metadata_file.exists():
        update_metadata(metadata_file, 
                       old_count=len(df_cleaned) + removed_count,
                       new_count=len(df_cleaned),
                       removed_count=removed_count)
    
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE!")
    print("="*80)
    print(f"📁 Cleaned files saved:")
    print(f"   📄 {output_csv.name}")
    print(f"   📄 {output_json.name}")
    print(f"\n💡 Original files kept as backup")
    print("="*80)

if __name__ == "__main__":
    main()
