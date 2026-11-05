'''
Uses Ashby and JobSpy to scrape jobs cleanly and reliably in headless mode. 
Outputs a JSON file with jobinformation and description as a full text wall.
'''

import json
import time
import requests
from bs4 import BeautifulSoup
from jobspy import scrape_jobs

def scrape_jobspy_clean(site_name, search_term, location, results_limit):
    """
    Scrapes jobs cleanly and reliably in headless mode.
    """
    print(f"[-] Scraping {site_name.capitalize()}...")
    try:
        df = scrape_jobs(
            site_name=[site_name],
            search_term=search_term,
            location=location,
            results_wanted=results_limit,
            hours_old=168,
            country_indeed='USA',
            linkedin_fetch_description=True # Pulls full long-form description
        )
        
        jobs = []
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                raw_description = row.get('description', '') or ''
                
                # Combine metadata with the full text wall
                metadata_header = (
                    f"=== JOB METADATA ===\n"
                    f"Source: {site_name.capitalize()}\n"
                    f"Title: {row.get('title', 'N/A')}\n"
                    f"Company: {row.get('company', 'N/A')}\n"
                    f"Location: {row.get('location', 'N/A')}\n"
                    f"Job Type / Workplace: {row.get('job_type', 'N/A')}\n"
                    f"Salary: {row.get('min_amount', '')} - {row.get('max_amount', '')} {row.get('currency', '')} ({row.get('interval', '')})\n"
                    f"URL: {row.get('job_url', 'N/A')}\n"
                    f"=====================\n\n"
                    f"=== FULL TEXT WALL ===\n"
                )
                
                full_text_wall = metadata_header + str(raw_description)

                job_entry = {
                    "company_name": str(row.get('company', 'Unknown Company')),
                    "title": str(row.get('title', 'Unknown Title')),
                    "url": str(row.get('job_url', '')),
                    "description": full_text_wall
                }
                jobs.append(job_entry)
        return jobs

    except Exception as e:
        print(f"[!] Error scraping {site_name}: {e}")
        return []

def scrape_ashby_clean(company_handles, search_term, results_limit):
    """
    Fetches clean text walls from Ashby ATS boards.
    """
    print("[-] Fetching Ashby boards...")
    jobs = []
    for org in company_handles:
        url = f"https://api.ashbyhq.com/posting-api/job-board/{org}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                postings = data.get("jobs", [])
                
                count = 0
                for post in postings:
                    title = post.get("title", "")
                    desc_html = post.get("descriptionHtml", "")
                    desc_plain = post.get("descriptionPlain", "")
                    
                    if desc_html:
                        soup = BeautifulSoup(desc_html, 'html.parser')
                        full_body_text = soup.get_text(separator="\n", strip=True)
                    else:
                        full_body_text = desc_plain

                    if search_term.lower() not in title.lower() and \
                       search_term.lower() not in full_body_text.lower():
                        continue
                        
                    if count >= results_limit:
                        break
                    
                    metadata_header = (
                        f"=== JOB METADATA ===\n"
                        f"Source: Ashby ({org.capitalize()})\n"
                        f"Title: {title}\n"
                        f"Company: {org.capitalize()}\n"
                        f"Department: {post.get('department', 'N/A')}\n"
                        f"Location: {post.get('location', 'N/A')}\n"
                        f"Workplace Type: {post.get('workplaceType', 'N/A')}\n"
                        f"URL: {post.get('jobUrl', '')}\n"
                        f"=====================\n\n"
                        f"=== FULL TEXT WALL ===\n"
                    )
                    
                    full_text_wall = metadata_header + full_body_text

                    jobs.append({
                        "company_name": org.capitalize(),
                        "title": title,
                        "url": post.get("jobUrl", ""),
                        "description": full_text_wall
                    })
                    count += 1
        except Exception as e:
            print(f"[!] Error fetching Ashby for {org}: {e}")
    return jobs

def main():
    search_keyword = "Software Engineer"
    location = "Remote"
    limit_per_site = 5
    
    all_jobs = []
    
    # 1. Scrape major boards
    for site in ["linkedin", "indeed", "glassdoor"]:
        site_jobs = scrape_jobspy_clean(site, search_keyword, location, limit_per_site)
        all_jobs.extend(site_jobs)
        time.sleep(2)
        
    # 2. Scrape Ashby companies
    ashby_companies = ["notion", "vercel", "linear"]
    ashby_jobs = scrape_ashby_clean(ashby_companies, search_keyword, limit_per_site)
    all_jobs.extend(ashby_jobs)
    
    # Save output
    output_filename = "jobs_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=4, ensure_ascii=False)
        
    print(f"\n[+] Saved {len(all_jobs)} jobs to '{output_filename}'")

if __name__ == "__main__":
    main()