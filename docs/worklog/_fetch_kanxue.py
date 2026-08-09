import urllib.request, re, sys

url = "https://bbs.kanxue.com/thread-287673.htm"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='replace')
    title = re.search(r'<title>(.*?)</title>', html)
    print("TITLE:", title.group(1) if title else "N/A")
    # Try different content div patterns
    for pattern in [
        r'<div class="postmessage[^"]*">(.*?)</div>',
        r'<td[^>]*class="t_f"[^>]*>(.*?)</td>',
        r'<div[^>]*id="postmessage_\d+"[^>]*>(.*?)</div>',
    ]:
        m = re.search(pattern, html, re.S)
        if m:
            text = re.sub(r'<[^>]+>', ' ', m.group(1))
            text = re.sub(r'&[a-z]+;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 100:
                print(text[:3000])
                break
    else:
        print("No content found. HTML length:", len(html))
except Exception as e:
    print(f"Error: {e}")
