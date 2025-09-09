import requests
import json
import re
from bs4 import BeautifulSoup


def fetch_repos(username):
    """Fetch all public repos for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def fetch_readme(owner, repo):
    """Fetch README content for a repo (if it exists)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.v3.raw"}  # Get raw README text
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return None


def clean_text(text):
    """Remove HTML tags and Markdown syntax from text."""
    if not text:
        return ""

    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator="\n")

    # Remove Markdown artifacts (headings, links, formatting)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links [text](url)
    text = re.sub(r"[*_`>#-]", "", text)  # *, _, `, #, >, -
    text = re.sub(r"\n{2,}", "\n\n", text)  # collapse extra newlines

    return text.strip()


def scrape_github_profile(username):
    repos = fetch_repos(username)
    projects_info = []

    for repo in repos:
        repo_name = repo["name"]
        description = repo.get("description", "")
        language = repo.get("language", "")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)

        readme_raw = fetch_readme(username, repo_name)
        readme_clean = clean_text(readme_raw) if readme_raw else "No README found."

        project_data = {
            "name": repo_name,
            "description": description,
            "language": language,
            "stars": stars,
            "forks": forks,
            "readme": readme_clean
        }
        projects_info.append(project_data)

    return {
        "username": username,
        "projects": projects_info
    }


if __name__ == "__main__":
    username = "thealphacubicle"  # 👈 replace with your GitHub username
    profile_data = scrape_github_profile(username)

    # Save results as JSON file
    import os
    os.makedirs("docs", exist_ok=True)
    with open(f"docs/{username}_projects.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Data saved to docs/{username}_projects.json")
