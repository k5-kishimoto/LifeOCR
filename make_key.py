import base64

# ↓ここを実際のJSONファイル名に変更してください
json_filename = "nth-facility-484301-p4-7d8f11b4d6df.json"

try:
    with open(json_filename, "rb") as f:
        # ファイルを読み込んでBase64に変換
        encoded = base64.b64encode(f.read()).decode('utf-8')
        
    print("--- 以下の文字列を .env に貼り付けてください ---")
    print(encoded)
    print("--- 終わり ---")
    print(f"文字数: {len(encoded)}")

except FileNotFoundError:
    print(f"エラー: {json_filename} が見つかりません。ファイル名を確認してください。")