import urllib.request
import re

folder_id = "1tGuzFz8jU79-4eDGmY5ZBqpMHU6G5Xd0"  # Ideation Phase
url = f"https://drive.google.com/drive/folders/{folder_id}"

req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
)

try:
    with urllib.request.urlopen(req) as res:
        html = res.read().decode('utf-8', errors='ignore')
        # Let's write the HTML to a file to inspect it
        with open("folder_page.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Successfully wrote folder page to folder_page.html")
        
        # Let's search for some patterns of file names or JSON objects
        # Often there is a list of items like: ["id", "name", "mimeType"] or similar
        print("Searching for title/name patterns:")
        # Look for occurrences of names and titles in JSON/script blocks
        # Usually, they are in blocks like: `_F_state` or `window.WIZ_global_data`
        for match in re.finditer(r'"([0-9a-zA-Z_\-\.]+)"', html):
            val = match.group(1)
            # Find any mention of common document types or .pdf, .docx, .xlsx, .txt, .csv, etc.
            if any(ext in val.lower() for ext in [".pdf", ".csv", ".xlsx", ".docx", ".png", ".jpg", ".txt", ".pptx", ".mp4", ".zip"]):
                print("Found file-like name:", val)
except Exception as e:
    print("Error fetching folder:", e)
