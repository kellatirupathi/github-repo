import streamlit as st
import requests
import os
import re
import json
import pandas as pd
import time
import threading
import queue
from datetime import datetime

# ========== CONFIGURATION ==========
# Directory to save extracted files
SAVE_DIRECTORY = "extracted_files"

# Create directory if it doesn't exist
try:
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create directory {SAVE_DIRECTORY}: {str(e)}")
    # Fall back to temp directory if needed
    import tempfile
    SAVE_DIRECTORY = tempfile.gettempdir()

# Gemini API Key - Default key used for all requests
GEMINI_API_KEY = "AIzaSyBBFKiwVjOlz06hGtjXe_NBa8D4Iyh_k_k"

# Use session state for storing repository content in memory when needed
if 'repo_contents' not in st.session_state:
    st.session_state.repo_contents = {}

# Configure Streamlit page
st.set_page_config(
    page_title="GitHub Repository Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ========== FUNCTION TO PARSE GITHUB REPO URL ==========
def parse_github_url(repo_url):
    """Extracts repo owner and name from a GitHub repository URL."""
    # Support for both https://github.com/owner/repo and github.com/owner/repo
    repo_url = repo_url.strip()
    if not repo_url.startswith("http"):
        repo_url = "https://" + repo_url
    
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url)
    if match:
        return match.group(1), match.group(2).split('/')[0]  # Handle potential trailing slashes or paths
    else:
        return None, None

# ========== FUNCTION TO FETCH REPO CONTENT ==========
def fetch_github_repo_text(repo_owner, repo_name, uuid):
    """Fetch all files from a GitHub repository and extract their content, excluding unnecessary files."""
    # Try main branch first, then master if main fails
    branches = ["main", "master"]
    data = None
    branch_used = None
    
    for branch in branches:
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            branch_used = branch
            break
    
    if not data:
        return None, "Failed to fetch repository structure. Repository might be private or doesn't exist."
    
    extracted_text = f"# Repository: {repo_owner}/{repo_name}\n"
    extracted_text += f"# Date Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Define files and extensions to exclude
    excluded_files = {"README.md", "favicon.ico", "pnpm-lock.yaml", "package-lock.json", ".gitignore"}
    excluded_extensions = (".png", ".jpg", ".jpeg", ".webp", ".svg", ".mp4", ".jar", ".ttf", ".json", 
                          ".ico", ".gif", ".woff", ".woff2", ".eot", ".map")
    excluded_dirs = ["node_modules", "dist", "build", "__pycache__", ".git", ".vscode", ".idea"]

    file_count = 0
    total_size = 0
    files_processed = []

    for item in data.get("tree", []):
        file_path = item["path"]
        
        # Skip excluded directories
        if any(excluded_dir in file_path for excluded_dir in excluded_dirs):
            continue
            
        # Skip excluded files and extensions
        if (file_path.split("/")[-1] in excluded_files or 
            file_path.lower().endswith(excluded_extensions)):
            continue

        if item["type"] == "blob":  # Process only files
            raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch_used}/{file_path}"
            
            # Fetch file content
            try:
                raw_response = requests.get(raw_url, timeout=10)
                if raw_response.status_code == 200:
                    content = raw_response.text
                    file_count += 1
                    total_size += len(content)
                    files_processed.append(file_path)
                    
                    # Add file header with path
                    extracted_text += f"\n\n{'=' * 80}\n"
                    extracted_text += f"FILE: {file_path}\n"
                    extracted_text += f"{'=' * 80}\n\n"
                    extracted_text += content
                    extracted_text += f"\n\n{'-' * 80}\n"
            except Exception as e:
                extracted_text += f"\n\nFailed to fetch: {file_path} - Error: {str(e)}\n\n"

    # Add summary at the top
    summary = f"# Summary:\n"
    summary += f"# - Files processed: {file_count}\n"
    summary += f"# - Total content size: {total_size} bytes\n"
    summary += f"# - Branch used: {branch_used}\n\n"
    summary += f"# Files included:\n# - " + "\n# - ".join(files_processed) + "\n\n"
    
    extracted_text = summary + extracted_text

    # Save extracted text to a file using UUID and also store in session state
    file_path = os.path.join(SAVE_DIRECTORY, f"{uuid}.txt")
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(extracted_text)
    except Exception as e:
        print(f"Warning: Could not write to file {file_path}: {str(e)}")
        # Store in session state as fallback
        st.session_state.repo_contents[uuid] = extracted_text
        return uuid, f"Successfully extracted {file_count} files from {repo_owner}/{repo_name} (stored in memory)"

    # Also store in session state as backup
    st.session_state.repo_contents[uuid] = extracted_text
    return file_path, f"Successfully extracted {file_count} files from {repo_owner}/{repo_name}"


# ========== FUNCTION TO ANALYZE TEXT WITH GEMINI ==========
def analyze_with_gemini(repo_text, assignment_text):
    """
    Use Gemini API to analyze repository content against an assignment question.
    """
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    **You are a skilled Assignment Evaluator who thinks like an experienced developer.** Your goal is to evaluate how well a given repository matches an assignment without being overly strict. Instead of just rule-based evaluation, consider the **practicality and intent behind the implementation**.

    ---

    ### **Step 1: Understand the Assignment**
    - Carefully **analyze the assignment requirements** to understand what the project is supposed to achieve.
    - Identify the **main functionalities**, expected features, and overall system design.
    - Consider **how an experienced developer would ideally implement this assignment** while allowing some flexibility in coding styles.

    ---

    ### **Step 2: Compare the Repository Against the Assignment**
    - Read through the repository code to check **if the required functionalities exist**.
    - Look for relevant logic, endpoints, functions, or database operations **that align with the assignment**.
    - If a feature is implemented **differently but still valid**, accept it as correct.
    - **If functionalities are missing but can be easily added, mention them constructively** rather than heavily penalizing.

    ---

    ### **Step 3: Handle Completely Unrelated Repositories**
    - If the repository content is **completely unrelated** to the assignment, **assign a 0% match immediately**.
        - **Example:** If the assignment is for a "Job Listing Platform," but the repo contains a "Weather Dashboard," the match percentage should be **0%**.
        - In this case, do not provide "achieved_functionalities" or "missing_functionalities" lists.
        - Simply explain why the repository is unrelated to the assignment.
    - If the repository contains **some coincidental but minor matching functionalities** (e.g., a login system that both projects might use), give a **very low score (1-5%)**.

    ---

    ### **Step 4: Provide a Fair and Constructive Evaluation**
    - Assign a **matching percentage** based on overall alignment.
    - **Do not reduce scores for minor coding style differences** unless they impact functionality.
    - Provide **constructive feedback** on:
        - **What is done well**
        - **What is missing**
        - **How to improve the project to match the assignment better**

    ---

    ### **Assignment to Evaluate:**
    {assignment_text}

    ---

    ### **Extracted Repository Content:**
    {repo_text[:100000]}  // Limiting to first 100,000 characters to avoid token limits

    ---

    ### **Expected JSON Output Format:**
    If the repository is COMPLETELY UNRELATED (0% match):
    ```json
    {{
        "match_percentage": "0%",
        "remarks": {{
            "overall_review": "A clear explanation of why the repository is completely unrelated to the assignment."
        }}
    }}
    ```

    Otherwise:
    ```json
    {{
        "match_percentage": "XX%",
        "remarks": {{
            "overall_review": "A balanced and constructive evaluation highlighting strengths and areas for improvement.",
            "achieved_functionalities": ["List of correctly implemented functionalities."],
            "missing_functionalities": ["List of functionalities missing or incomplete as per the assignment."]
        }}
    }}
    ```

    IMPORTANT: Your response should ONLY contain the valid JSON without any additional text, explanation, or markdown formatting.
    """

    # API request payload
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    headers = {"Content-Type": "application/json"}

    # Make request to Gemini API
    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            json_response = response.json()
            if "candidates" in json_response and json_response["candidates"]:
                # Extract the raw text response
                raw_response = json_response["candidates"][0]["content"]["parts"][0]["text"]
                
                # Clean up the response to extract just the JSON
                # Remove markdown code blocks if present
                clean_response = raw_response.replace("```json", "").replace("```", "").strip()
                
                # Try to parse as JSON to validate
                try:
                    parsed_json = json.loads(clean_response)
                    # Return the properly formatted JSON string
                    return json.dumps(parsed_json, indent=2), parsed_json
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON response from Gemini API: {clean_response}", None
            else:
                return "Error: No meaningful response from Gemini API", None
        else:
            return f"Error calling Gemini API: {response.status_code} - {response.text}", None
    except Exception as e:
        return f"Error during API call: {str(e)}", None


# ========== PROCESS SINGLE REPOSITORY ==========
def process_single_repository(repo_url, uuid, assignment_text, status_placeholder, result_placeholder):
    """Process a single repository - extract and analyze."""
    if not uuid or not repo_url or not assignment_text:
        status_placeholder.error("Please fill all required fields.")
        return False
        
    # 1. Parse URL
    repo_owner, repo_name = parse_github_url(repo_url)
    if not repo_owner or not repo_name:
        status_placeholder.error(f"Invalid GitHub repository URL: {repo_url}")
        return False
        
    # 2. Extract repository
    status_placeholder.info(f"Extracting content from {repo_owner}/{repo_name}...")
    file_path, message = fetch_github_repo_text(repo_owner, repo_name, uuid)
    
    if not file_path:
        status_placeholder.error(message)
        return False
        
    status_placeholder.success(f"Repository extracted: {repo_owner}/{repo_name}")
    
    # 3. Analyze
    status_placeholder.info(f"Analyzing {repo_owner}/{repo_name} against assignment requirements...")
    with open(file_path, "r", encoding="utf-8") as f:
        extracted_text = f.read()
    
    analysis_text, analysis_json = analyze_with_gemini(extracted_text, assignment_text)
    
    if analysis_json:
        display_analysis_result(analysis_json, repo_owner, repo_name, result_placeholder)
        status_placeholder.success(f"Analysis complete for {repo_owner}/{repo_name}")
        return True
    else:
        status_placeholder.error(f"Analysis failed for {repo_owner}/{repo_name}: {analysis_text}")
        return False


# ========== THREAD WORKER FOR BATCH PROCESSING ==========
def process_repo_worker(task_queue, result_queue, assignment_text):
    """Worker thread to process repository tasks."""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to signal thread termination
            break
            
        repo_url, uuid = task
        repo_owner, repo_name = parse_github_url(repo_url)
        
        # Skip invalid URLs
        if not repo_owner or not repo_name:
            result_queue.put({
                "repo_url": repo_url,
                "uuid": uuid,
                "status": "error",
                "message": "Invalid GitHub repository URL",
                "match_percentage": None,
                "analysis": None
            })
            task_queue.task_done()
            continue
            
        # Extract repository
        try:
            file_path, message = fetch_github_repo_text(repo_owner, repo_name, uuid)
            
            if not file_path:
                result_queue.put({
                    "repo_url": repo_url,
                    "uuid": uuid,
                    "status": "error",
                    "message": message,
                    "match_percentage": None,
                    "analysis": None
                })
                task_queue.task_done()
                continue
                
            # Try to read from file first, then fallback to session state
            extracted_text = ""
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            elif uuid in st.session_state.repo_contents:
                extracted_text = st.session_state.repo_contents[uuid]
            else:
                result_queue.put({
                    "repo_url": repo_url,
                    "uuid": uuid,
                    "status": "error",
                    "message": "Repository content not found",
                    "match_percentage": None,
                    "analysis": None
                })
                task_queue.task_done()
                continue
                
            analysis_text, analysis_json = analyze_with_gemini(extracted_text, assignment_text)
            
            if analysis_json:
                result_queue.put({
                    "repo_url": repo_url,
                    "uuid": uuid,
                    "status": "success",
                    "message": "Analysis complete",
                    "match_percentage": analysis_json.get("match_percentage", "N/A"),
                    "analysis": analysis_json
                })
            else:
                result_queue.put({
                    "repo_url": repo_url,
                    "uuid": uuid,
                    "status": "error",
                    "message": f"Analysis failed: {analysis_text}",
                    "match_percentage": None,
                    "analysis": None
                })
        except Exception as e:
            result_queue.put({
                "repo_url": repo_url,
                "uuid": uuid,
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "match_percentage": None,
                "analysis": None
            })
        finally:
            task_queue.task_done()


# ========== BATCH PROCESS MULTIPLE REPOSITORIES ==========
def process_batch_repositories(repo_data, assignment_text, progress_bar, status_placeholder, results_container):
    """Process multiple repositories in parallel using a thread pool."""
    if not repo_data or not assignment_text:
        status_placeholder.error("Please provide repository data and assignment text.")
        return
        
    num_repositories = len(repo_data)
    status_placeholder.info(f"Processing {num_repositories} repositories...")
    
    # Create queues
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Number of worker threads (adjust based on needs)
    num_workers = min(4, num_repositories)  # Maximum 4 workers to avoid API rate limits
    
    # Create and start worker threads
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=process_repo_worker,
            args=(task_queue, result_queue, assignment_text),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Add tasks to the queue
    for repo in repo_data:
        task_queue.put((repo["repo_url"], repo["uuid"]))
    
    # Add termination signals for threads    
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Create placeholder for results table
    results_df = pd.DataFrame(columns=["UUID", "Repository", "Match %", "Status"])
    results_table = results_container.empty()
    
    # Track progress and update results in real-time
    completed = 0
    results = []
    
    # Monitor the result queue until all tasks are done
    while completed < num_repositories:
        try:
            result = result_queue.get(timeout=0.5)
            results.append(result)
            completed += 1
            
            # Update progress bar
            progress_bar.progress(completed / num_repositories)
            
            # Update results table
            results_df = pd.DataFrame([
                {
                    "UUID": r.get("uuid", ""),
                    "Repository": r.get("repo_url", ""),
                    "Match %": r.get("match_percentage", "Error"),
                    "Status": "✅ Success" if r.get("status") == "success" else f"❌ {r.get('message', 'Failed')}"
                }
                for r in results
            ])
            results_table.dataframe(results_df)
            
        except queue.Empty:
            # No results available yet
            continue
    
    # Wait for threads to finish
    for thread in threads:
        thread.join()
        
    progress_bar.progress(1.0)
    status_placeholder.success(f"Completed processing {num_repositories} repositories")
    
    # Return final results for further processing
    return results


# ========== FUNCTION TO DISPLAY ANALYSIS RESULT ==========
def display_analysis_result(result_json, repo_owner, repo_name, container, inside_expander=False):
    """Display the analysis result in a structured format."""
    container.markdown(f"## Analysis Results for [{repo_owner}/{repo_name}](https://github.com/{repo_owner}/{repo_name})")
    
    # Display match percentage prominently
    container.markdown(f"### Match Percentage: **{result_json.get('match_percentage', 'N/A')}**")
    
    # Display remarks
    if 'remarks' in result_json:
        remarks = result_json['remarks']
        
        container.markdown("### Overall Review")
        container.write(remarks.get('overall_review', 'No review provided'))
        
        # Only show achieved and missing if they exist (not for 0% matches)
        if 'achieved_functionalities' in remarks:
            container.markdown("### Achieved Functionalities")
            for item in remarks.get('achieved_functionalities', []):
                container.markdown(f"- ✅ {item}")
        
        if 'missing_functionalities' in remarks:
            container.markdown("### Missing or Incomplete Functionalities")
            for item in remarks.get('missing_functionalities', []):
                container.markdown(f"- ❌ {item}")
    
    # Show raw JSON differently based on whether we're already in an expander
    if inside_expander:
        # If already in an expander, just show the JSON directly
        container.markdown("### Raw JSON")
        container.json(result_json)
    else:
        # If not in an expander, we can use an expander
        with container.expander("View Raw JSON"):
            container.json(result_json)


# ========== UI: TABS FOR SINGLE AND BATCH MODES ==========
def main():
    st.title("🔍 GitHub Repository Analyzer")
    st.markdown("""
    This tool extracts code from GitHub repositories and analyzes them against an assignment question using Gemini AI.
    """)
    
    # Create tabs for single and batch modes
    tab1, tab2 = st.tabs(["Single Repository", "Multiple Repositories"])
    
    # === SINGLE REPOSITORY TAB ===
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Use session state to persist values
            if 'single_uuid' not in st.session_state:
                st.session_state.single_uuid = ""
                
            single_uuid = st.text_input(
                "Enter UUID (for file storage):",
                key="single_uuid_input",
                value=st.session_state.single_uuid,
                help="A unique identifier to keep track of your analysis."
            )
            st.session_state.single_uuid = single_uuid
            
            single_repo_url = st.text_input(
                "GitHub Repository URL:",
                key="single_repo_url",
                help="Example: https://github.com/username/repository"
            )
            
            with st.expander("Advanced Options"):
                st.checkbox("Include README files", value=False, key="single_include_readme")
                st.checkbox("Include JSON files", value=False, key="single_include_json")
        
        with col2:
            single_assignment = st.text_area(
                "Assignment Question/Requirements:",
                key="single_assignment",
                height=200,
                help="Paste the complete assignment description here."
            )
        
        single_col1, single_col2, single_col3 = st.columns([1, 1, 2])
        
        with single_col1:
            single_extract_btn = st.button("1. Extract Repository", key="single_extract", type="primary")
            
        with single_col2:
            single_analyze_btn = st.button("2. Analyze", key="single_analyze", type="primary")
            
        with single_col3:
            single_extract_analyze_btn = st.button("Extract & Analyze", key="single_both", type="primary")
            
        # Status and results containers
        single_status = st.empty()
        single_results = st.container()
        
        # Handle single repository actions
        if single_extract_btn:
            if not single_uuid or not single_repo_url:
                single_status.error("Please provide both UUID and repository URL.")
            else:
                repo_owner, repo_name = parse_github_url(single_repo_url)
                if not repo_owner or not repo_name:
                    single_status.error("Invalid GitHub repository URL.")
                else:
                    single_status.info(f"Extracting repository content from {repo_owner}/{repo_name}...")
                    file_path, message = fetch_github_repo_text(repo_owner, repo_name, single_uuid)
                    
                    if file_path:
                        single_status.success(message)
                    else:
                        single_status.error(message)
                        
        if single_analyze_btn:
            if not single_uuid or not single_assignment:
                single_status.error("Please provide UUID and assignment text.")
            else:
                # Check if file exists
                file_path = os.path.join(SAVE_DIRECTORY, f"{single_uuid}.txt")
                if not os.path.exists(file_path):
                    single_status.error(f"No repository content found for UUID: {single_uuid}. Please extract the repository first.")
                else:
                    # Extract repo owner/name from the file content for display
                    with open(file_path, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        repo_match = re.match(r"# Repository: ([^/]+)/([^/\n]+)", first_line)
                        if repo_match:
                            repo_owner, repo_name = repo_match.groups()
                        else:
                            repo_owner, repo_name = "Unknown", "Repository"
                    
                    single_status.info(f"Analyzing repository with Gemini API...")
                    with open(file_path, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                    
                    analysis_text, analysis_json = analyze_with_gemini(extracted_text, single_assignment)
                    
                    if analysis_json:
                        display_analysis_result(analysis_json, repo_owner, repo_name, single_results)
                        single_status.success("Analysis complete!")
                    else:
                        single_status.error(f"Analysis failed: {analysis_text}")
        
        if single_extract_analyze_btn:
            process_single_repository(
                single_repo_url, 
                single_uuid, 
                single_assignment, 
                single_status, 
                single_results
            )
    
    # === MULTIPLE REPOSITORIES TAB ===
    with tab2:
        st.markdown("### Batch Process Multiple Repositories")
        
        # Instructions
        st.info("""
        You can analyze multiple repositories at once. Please provide your data in one of these formats:
        1. CSV file with columns: uuid,repo_url
        2. Enter data manually in the text area below (one repo per line: uuid,repo_url)
        """)
        
        # Input methods
        batch_tab1, batch_tab2 = st.tabs(["Upload CSV", "Enter Manually"])
        
        with batch_tab1:
            uploaded_file = st.file_uploader("Upload CSV file (uuid,repo_url)", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = ["uuid", "repo_url"]
                    
                    if all(col in df.columns for col in required_columns):
                        st.success(f"Successfully loaded {len(df)} repositories")
                        st.dataframe(df[required_columns], hide_index=True)
                        
                        # Store in session state
                        st.session_state.batch_data = df[required_columns].to_dict('records')
                    else:
                        st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
            
        with batch_tab2:
            manual_data = st.text_area(
                "Enter repositories (one per line in format: uuid,repo_url)",
                height=150,
                help="Example:\nstudent1,https://github.com/user1/repo1\nstudent2,https://github.com/user2/repo2"
            )
            
            if manual_data.strip():
                try:
                    # Parse manual input
                    repos = []
                    for line in manual_data.strip().split('\n'):
                        if ',' in line:
                            parts = line.split(',', 1)
                            uuid = parts[0].strip()
                            repo_url = parts[1].strip()
                            repos.append({"uuid": uuid, "repo_url": repo_url})
                    
                    if repos:
                        st.success(f"Successfully parsed {len(repos)} repositories")
                        st.dataframe(pd.DataFrame(repos), hide_index=True)
                        
                        # Store in session state
                        st.session_state.batch_data = repos
                    else:
                        st.warning("No valid repository data found")
                except Exception as e:
                    st.error(f"Error parsing input: {str(e)}")
        
        # Assignment text for batch processing
        batch_assignment = st.text_area(
            "Assignment Question/Requirements for all repositories:",
            key="batch_assignment",
            height=200
        )
        
        # Process button
        batch_process_btn = st.button("Process All Repositories", type="primary")
        
        # Containers for batch processing status and results
        batch_progress = st.empty()
        batch_status = st.empty()
        batch_results = st.container()
        
        if batch_process_btn:
            if not hasattr(st.session_state, 'batch_data') or not st.session_state.batch_data:
                batch_status.error("No repository data provided. Please upload a CSV or enter data manually.")
            elif not batch_assignment:
                batch_status.error("Please provide the assignment text.")
            else:
                # Setup progress tracking
                progress_bar = batch_progress.progress(0.0)
                
                # Process all repositories
                results = process_batch_repositories(
                    st.session_state.batch_data,
                    batch_assignment,
                    progress_bar,
                    batch_status,
                    batch_results
                )
                
                if results:
                    # Create a summary table
                    st.markdown("### Summary of Results")
                    
                    # Count by match percentage ranges
                    match_ranges = {
                        "0%": 0,
                        "1-25%": 0,
                        "26-50%": 0,
                        "51-75%": 0,
                        "76-100%": 0,
                        "Error": 0
                    }
                    
                    for result in results:
                        if result['status'] != 'success':
                            match_ranges["Error"] += 1
                            continue
                            
                        match_str = result.get('match_percentage', '0%')
                        try:
                            # Handle percentage format (e.g., "75%")
                            match_val = float(match_str.strip('%'))
                            
                            if match_val == 0:
                                match_ranges["0%"] += 1
                            elif match_val <= 25:
                                match_ranges["1-25%"] += 1
                            elif match_val <= 50:
                                match_ranges["26-50%"] += 1
                            elif match_val <= 75:
                                match_ranges["51-75%"] += 1
                            else:
                                match_ranges["76-100%"] += 1
                        except ValueError:
                            match_ranges["Error"] += 1
                    
                    # Display summary as a horizontal bar chart
                    summary_data = pd.DataFrame({
                        "Range": list(match_ranges.keys()),
                        "Count": list(match_ranges.values())
                    })
                    
                    st.bar_chart(summary_data, x="Range", y="Count")
                    
                    # Option to download detailed results
                    detailed_results = []
                    for result in results:
                        if result['status'] != 'success' or not result['analysis']:
                            detailed_results.append({
                                "UUID": result['uuid'],
                                "Repository": result['repo_url'],
                                "Match Percentage": "Error",
                                "Status": result['message'],
                                "Full Remarks": json.dumps({})  # Empty JSON for errors
                            })
                        else:
                            # Include the full remarks JSON
                            full_remarks = json.dumps(result['analysis']['remarks'])
                            
                            detailed_results.append({
                                "UUID": result['uuid'],
                                "Repository": result['repo_url'],
                                "Match Percentage": result['match_percentage'],
                                "Status": "Success",
                                "Full Remarks": full_remarks
                            })
                    
                    results_df = pd.DataFrame(detailed_results)
                    
                    # Offer download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Detailed Results CSV",
                        data=csv,
                        file_name="github_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Display individual results in expandable sections
                    st.markdown("### Detailed Results")
                    for result in results:
                        if result['status'] == 'success' and result['analysis']:
                            with st.expander(f"{result['uuid']} - {result['repo_url']} ({result['match_percentage']})"):
                                # Extract repo owner/name for display
                                repo_owner, repo_name = parse_github_url(result['repo_url'])
                                if not repo_owner:
                                    repo_owner, repo_name = "Unknown", "Repository"
                                
                                display_analysis_result(result['analysis'], repo_owner, repo_name, st, inside_expander=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### How to use this tool")
    
    with st.expander("Single Repository Mode"):
        st.markdown("""
        1. Enter a unique identifier (UUID) to keep track of your analysis
        2. Paste the GitHub repository URL you want to analyze
        3. Paste the assignment requirements
        4. Click "Extract & Analyze" to process the repository
        """)
    
    with st.expander("Multiple Repositories Mode"):
        st.markdown("""
        1. Upload a CSV file with columns: `uuid,repo_url` OR enter data manually
        2. Paste the assignment requirements (same for all repositories)
        3. Click "Process All Repositories" to analyze all repositories
        4. View the summary and download detailed results
        """)
    
    st.markdown("---")
    st.markdown("### About File Storage")
    st.info("""
    **Important information about file storage:**
    
    1. When deployed to Streamlit Cloud, file storage may be ephemeral, meaning files could be lost when the app restarts.
    
    2. This application uses both:
       - Local file storage (when possible)
       - In-memory storage through session state (as backup)
       
    3. In deployed environments:
       - Repository data may not persist between sessions
       - It's recommended to extract and analyze repositories in the same session
       - For large batch processing, consider downloading results immediately
    """)
    
    st.markdown("---")
    st.markdown("**Note:** Analysis results are cached by UUID. If you re-analyze with the same UUID, the repository content won't be re-extracted unless you explicitly use the extraction button.")

# Run the main application
if __name__ == "__main__":
    main()