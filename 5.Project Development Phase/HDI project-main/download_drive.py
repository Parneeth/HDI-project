import os
import urllib.request
import re

files = {
    "file1_10YJycxKt": "10YJycxKtStR65DZwACrWvULoHZfTd5OB",
    "file2_1OHpL6Ah": "1OHpL6Ah0uvQ_5rWrLz8TcnUvAPJiJtrN",
    "file3_1_DHNoQ9": "1_DHNoQ90q5_c2WX-vwpMZuJodRdsPaDr",
    "file4_1UF8Xwi": "1UF8XwiS-C_XHXI-2TjkpTULE6o4q-Zbk",
    "file5_1AiwFY7": "1AiwFY7sNLtb6Wd5FSF68gVyht8wpKVjU",
    "file6_1tGuzFz": "1tGuzFz8jU79-4eDGmY5ZBqpMHU6G5Xd0",
}

os.makedirs("gdrive_downloads", exist_ok=True)

def download_file(file_id, output_path):
    # Try direct download URL first
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            # If Google Drive shows a warning / confirm page, we need to handle that.
            # But let's check what the content starts with.
            if b"confirm=" in content or b"Google Drive - Virus scan warning" in content:
                # Extract confirm token if present
                html = content.decode('utf-8', errors='ignore')
                match = re.search(r'confirm=([0-9A-Za-z_]+)', html)
                if match:
                    token = match.group(1)
                    confirm_url = f"https://docs.google.com/uc?export=download&confirm={token}&id={file_id}"
                    confirm_req = urllib.request.Request(
                        confirm_url,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(confirm_req) as confirm_response:
                        content = confirm_response.read()
            
            with open(output_path, "wb") as f:
                f.write(content)
            print(f"Downloaded {file_id} to {output_path} (size: {len(content)} bytes)")
            # Print first 200 chars if text
            try:
                print(f"Head of {output_path}:")
                print(content[:500].decode('utf-8', errors='ignore'))
            except Exception as e:
                print("Could not decode head as utf-8:", e)
    except Exception as e:
        print(f"Failed to download {file_id}: {e}")

for name, fid in files.items():
    print(f"\n--- Downloading {name} ({fid}) ---")
    download_file(fid, f"gdrive_downloads/{name}")
