import urllib.request
import re
import json

folders = {
    "1. Ideation Phase": "1tGuzFz8jU79-4eDGmY5ZBqpMHU6G5Xd0",
    "2. Requirement Analysis": "1AiwFY7sNLtb6Wd5FSF68gVyht8wpKVjU",
    "3. Project Design Phase": "1UF8XwiS-C_XHXI-2TjkpTULE6o4q-Zbk",
    "4. Project Planning Phase": "1_DHNoQ90q5_c2WX-vwpMZuJodRdsPaDr",
    "5. Project Development Phase": "1OHpL6Ah0uvQ_5rWrLz8TcnUvAPJiJtrN",
    "6. Project Documentation": "10YJycxKtStR65DZwACrWvULoHZfTd5OB",
}

def scan_folder(folder_name, folder_id):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    try:
        with urllib.request.urlopen(req) as res:
            html = res.read().decode('utf-8', errors='ignore')
            
            # Find all file/folder structures in the state.
            # In Drive folders, WIZ_global_data or some other state has lists of files.
            # Let's search for IDs (28-44 chars starting with 1) and titles next to them.
            # We can also look for filenames with extensions
            matches = re.findall(r'["\'](1[0-9a-zA-Z_\-]{32,40})["\'],\s*["\']([^"\']+\.[a-zA-Z0-9]{2,4})["\']', html)
            
            # Let's also do a broader scan for any PDF, CSV, PNG, etc.
            broader_matches = re.findall(r'["\'](1[a-zA-Z0-9_\-]{32,40})["\'].*?["\']([^"\']+?\.(?:pdf|csv|xlsx|docx|pptx|png|jpg|txt|zip|json|md))["\']', html)
            
            # Also, look for items that are google docs, sheets, slides (no extension in title, but mimeType is present)
            # A typical item looks like: ["1id", "Title of Doc", "application/vnd.google-apps.document"]
            # Let's search for this pattern:
            gdoc_matches = re.findall(r'["\'](1[a-zA-Z0-9_\-]{32,40})["\'],\s*["\']([^"\']+?)["\'],\s*["\'](application/vnd\.google-apps\.[a-z\-]+)["\']', html)
            
            all_found = {}
            for fid, fname in matches:
                all_found[fid] = (fname, "Binary File / Asset")
            for fid, fname in broader_matches:
                all_found[fid] = (fname, "Binary File / Asset")
            for fid, fname, mime in gdoc_matches:
                all_found[fid] = (fname, mime)
                
            print(f"\n📁 {folder_name} ({folder_id}):")
            if not all_found:
                # If nothing found, print some snippet of strings that look like titles to debug
                # Let's look for any strings with uppercase and lowercase words
                word_matches = re.findall(r'["\']([A-Z][a-zA-Z0-9_\s\-]{5,50})["\']', html)
                print("  No files with standard formats detected directly.")
                print("  Some text titles found in page:")
                unique_words = list(set(word_matches))[:8]
                for w in unique_words:
                    print(f"    - {w}")
            else:
                for fid, (fname, mime) in all_found.items():
                    print(f"  📄 {fname} (ID: {fid}, Type: {mime})")
                    
    except Exception as e:
        print(f"Error scanning {folder_name}: {e}")

for name, fid in folders.items():
    scan_folder(name, fid)
