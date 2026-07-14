import urllib.request
import re
import json

urls = {
    "file1": "https://drive.google.com/file/d/10YJycxKtStR65DZwACrWvULoHZfTd5OB/view",
    "file2": "https://drive.google.com/file/d/1OHpL6Ah0uvQ_5rWrLz8TcnUvAPJiJtrN/view",
    "file3": "https://drive.google.com/file/d/1_DHNoQ90q5_c2WX-vwpMZuJodRdsPaDr/view",
    "file4": "https://drive.google.com/file/d/1UF8XwiS-C_XHXI-2TjkpTULE6o4q-Zbk/view",
    "file5": "https://drive.google.com/file/d/1AiwFY7sNLtb6Wd5FSF68gVyht8wpKVjU/view",
    "file6": "https://drive.google.com/file/d/1tGuzFz8jU79-4eDGmY5ZBqpMHU6G5Xd0/view",
}

for name, url in urls.items():
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    try:
        with urllib.request.urlopen(req) as res:
            html = res.read().decode('utf-8', errors='ignore')
            # Look for itemJson in the HTML
            # e.g., itemJson: [null,"6. Project Documentation",..., "application/vnd.google-apps.folder"]
            match = re.search(r'itemJson:\s*(\[.*?\]);</script>', html)
            if match:
                json_str = match.group(1)
                # Parse or clean up json_str. It can have nulls or be slightly malformed for standard JSON,
                # so we can use eval with safe globals or regex to extract name and type
                # Simple regex extraction is safer
                title_match = re.search(r'itemJson:\s*\[null,"([^"]+)"', html)
                mime_match = re.search(r'"(application/[^"]+|image/[^"]+|text/[^"]+)"', html)
                
                title = title_match.group(1) if title_match else "Unknown Title"
                mime = mime_match.group(1) if mime_match else "Unknown Mime"
                
                print(f"{name} ({url.split('/')[-2]}):")
                print(f"  Title: {title}")
                print(f"  Type: {mime}")
            else:
                print(f"{name} ({url.split('/')[-2]}): Could not find itemJson")
                # print first few lines of html
                print("HTML snippet:", html[:500])
    except Exception as e:
        print(f"Failed {name}: {e}")
