import requests
import time

mir_ip = "192.168.50.26"
host = f"http://{mir_ip}/api/v2.0.0/"
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Basic RGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
}

# 你要依序執行的任務名稱清單
target_missions = ["back2desk",]  # ← 替換成你自己的任務名稱

# 取得所有任務資料
res = requests.get(host + "missions", headers=headers)
missions = res.json()

for target_name in target_missions:
    mission = next((m for m in missions if m['name'] == target_name), None)

    if not mission:
        print(f"⚠️ 找不到任務名稱: {target_name}")
        continue

    print(f"\n▶️ 加入任務: {mission['name']} ({mission['guid']})")
    response = requests.post(
        url=host + "mission_queue",
        headers=headers,
        json={"mission_id": mission['guid']}
    )
    print("  狀態碼:", response.status_code)
    print("  回傳:", response.text)

    if response.status_code == 201:
        mission_queue = response.json()
        mission_id = mission_queue.get('id')

        print("⌛ 等待機器人完成任務...")
        while True:
            status_res = requests.get(host + "mission_queue/" + str(mission_id), headers=headers)
            status_data = status_res.json()
            state = status_data.get("state")

            if state == "Done":
                print("✅ 任務完成:", target_name)
                break
            elif state == "Failed":
                print("❌ 任務失敗:", target_name)
                break
            else:
                print("⏳ 執行中:", state)
                time.sleep(2)
    else:
        print(f"❌ 無法送出任務 {target_name}")

print("\n🏁 所有任務處理完畢。")
