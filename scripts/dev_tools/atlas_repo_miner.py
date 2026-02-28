#!/usr/bin/env python3
"""
atlas_repo_miner.py - A flexible CLI tool for scanning and cloning AI & Materials Science GitHub repositories.

Features:
  --mode [target|global|both]:
     'target' scans specific GitHub users or organizations.
     'global' searches GitHub globally using topic/keyword queries.
  --limit N: Max number of repos to find and clone (default: 50).
  --min-stars N: Minimum stars required to be considered (default: 10).
  --clone / --no-clone: Whether to automatically clone the found repos to references/

This tool respects GitHub API rate limits.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request

# Configuration
TARGETS = [
    ("users", "frostedoyster"), ("users", "jchodera"), ("users", "ghliu"),
    ("users", "bowen-bd"), ("users", "recisic"), ("users", "lan496"),
    ("orgs", "materialsproject"), ("users", "junwoony"), ("users", "tsmathis")
]

GLOBAL_QUERIES = [
    "topic:materials-science topic:machine-learning",
    "topic:materials-informatics",
    "machine learning interatomic potential",
    "dft machine learning",
    "ab initio machine learning materials",
    "equivariant neural network materials",
    "active learning materials discovery"
]

RELEVANCE_KEYWORDS = [
    "material", "crystal", "molecular", "dynamics", "md", "dft", "quantum",
    "gnn", "graph", "equivariant", "topology", "active learning", "bayesian",
    "potential", "mlip", "alloy", "thermodynamic", "mace", "polymer",
    "cheminformatics", "physics", "ab initio", "force field", "matbench",
    "property", "chgnet", "matgl", "m3gnet", "cgcnn", "reaction-network"
]

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ATLAS-RepoMiner/1.0'})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
        return None

def is_relevant(repo_name, repo_desc, repo_topics):
    text = (str(repo_name) + " " + str(repo_desc) + " " + " ".join(repo_topics or [])).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)

def search_targets(min_stars):
    print("[*] Scanning specific targets...")
    all_repos = []
    for type_str, name in TARGETS:
        print(f"    -> Fetching repos for {name}...")
        page = 1
        while True:
            url = f"https://api.github.com/{type_str}/{name}/repos?per_page=100&page={page}"
            data = fetch_json(url)
            if not data:
                break
            for repo in data:
                if repo.get("stargazers_count", 0) >= min_stars:
                    if is_relevant(repo["name"], repo["description"], repo.get("topics", [])):
                        all_repos.append(repo)
            if len(data) < 100:
                break
            page += 1
            time.sleep(1)
        time.sleep(1)
    return all_repos

def search_global(min_stars):
    print("[*] Searching GitHub globally...")
    all_items = {}
    for q in GLOBAL_QUERIES:
        query_encoded = urllib.parse.quote(q)
        url = f"https://api.github.com/search/repositories?q={query_encoded}&sort=stars&order=desc&per_page=100"
        print(f"    -> Query: {q}")

        data = fetch_json(url)
        if data and 'items' in data:
            for item in data['items']:
                if item.get("stargazers_count", 0) >= min_stars:
                    all_items[item['full_name']] = item
        time.sleep(2) # Respect rate limits
    return list(all_items.values())

def clone_repos(repos, num_to_clone, out_dir="references"):
    os.makedirs(out_dir, exist_ok=True)

    # Check existing to avoid redundant cloning attempts
    existing_dirs = set(os.listdir(out_dir))

    to_clone = []
    for r in repos:
        if r['name'] not in existing_dirs:
            to_clone.append(r)
            if len(to_clone) >= num_to_clone:
                break

    print(f"\n[*] Selected {len(to_clone)} new repositories to clone.")

    original_cwd = os.getcwd()
    os.chdir(out_dir)

    try:
        for i, r in enumerate(to_clone, 1):
            owner = r["owner"]["login"]
            name = r["name"]
            url = r["html_url"]
            stars = r["stargazers_count"]

            print(f"[{i}/{len(to_clone)}] Cloning {owner}/{name} (Stars: {stars})...")

            os.makedirs(owner, exist_ok=True)
            target_dir = os.path.join(owner, name)

            if not os.path.exists(target_dir):
                subprocess.run(["git", "clone", "--depth", "1", url, target_dir])

    finally:
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(description="ATLAS Repo Miner")
    parser.add_argument("--mode", choices=["target", "global", "both"], default="both", help="Search mode")
    parser.add_argument("--limit", type=int, default=50, help="Max number of repos to clone")
    parser.add_argument("--min-stars", type=int, default=10, help="Minimum stars")
    parser.add_argument("--no-clone", action="store_true", help="Skip cloning, just save to JSON")

    args = parser.parse_args()

    repos = []
    if args.mode in ["target", "both"]:
        repos.extend(search_targets(args.min_stars))

    if args.mode in ["global", "both"]:
        repos.extend(search_global(args.min_stars))

    # Deduplicate by full_name
    unique_repos = {r['full_name']: r for r in repos}
    sorted_repos = sorted(list(unique_repos.values()), key=lambda x: x['stargazers_count'], reverse=True)

    print(f"\n[+] Total unique relevant repos found: {len(sorted_repos)}")

    os.makedirs("references", exist_ok=True)
    out_file = "references/miner_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sorted_repos, f, indent=2, ensure_ascii=False)
    print(f"[+] Results saved to {out_file}")

    if not args.no_clone and args.limit > 0:
        clone_repos(sorted_repos, args.limit)

if __name__ == "__main__":
    main()
